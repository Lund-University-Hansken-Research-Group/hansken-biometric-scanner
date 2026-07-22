import json
import os
from pathlib import Path

import numpy as np
from pickle import Unpickler

from hansken_extraction_plugin.api.extraction_plugin import ExtractionPlugin
from hansken_extraction_plugin.api.plugin_info import Author, MaturityLevel, PluginId, PluginInfo
from logbook import Logger

log = Logger(__name__)


DEFAULT_EXTENSIONS = {
    'faceRecognition': {'.pkl', '.pickle', '.dat', '.pb', '.pt', '.pth', '.h5', '.caffemodel', '.uff'},
    'voiceBiometric': {'.pt', '.pth', '.pb', '.h5', '.wav', '.m4a', '.mp3', '.pkl', '.pickle'},
    'embeddingCache': {'.npy', '.npz', '.pkl', '.pickle', '.json', '.csv'},
}

DEFAULT_PATTERNS = {
    'faceRecognition': ['face', 'facial', 'dlib', 'facenet', 'arcface', 'retina', 'vgg', 'embedding', 'encodings'],
    'voiceBiometric': ['voice', 'speaker', 'voiceprint', 'xvector', 'speech', 'audio_embedding'],
    'embeddingCache': ['embedding', 'embeddings', 'encoding', 'encodings'],
}

MODEL_FRAMEWORKS = {
    'dlib': ['dlib', 'shape_predictor'],
    'tensorflow': ['tensorflow', 'tf_', '.pb'],
    'pytorch': ['pytorch', 'torch', '.pt', '.pth'],
    'keras': ['keras', '.h5'],
    'onnx': ['onnx', '.onnx'],
    'scikit': ['sklearn', 'scikit'],
}

FACE_EMBEDDING_DIMS = [128, 256, 512, 1024]

BIOMETRIC_CONTENT_KEYWORDS = [
    'encodings', 'embeddings', 'faces', 'face_encodings',
    'face_embeddings', 'facevector', 'facenet', 'dlib_encodings',
]

NAME_CONTENT_KEYWORDS = [
    'names', 'labels', 'ids', 'identities', 'persons', 'people', 'subjects',
]


def load_patterns():
    patterns_path = os.environ.get('BIOMETRIC_PATTERNS_PATH', '/app/patterns.json')
    try:
        with open(patterns_path, 'r') as f:
            custom = json.load(f)
        log.info(f'Loaded custom patterns from {patterns_path}')
        return custom
    except FileNotFoundError:
        log.debug(f'No custom patterns found at {patterns_path}, using defaults')
        return None
    except json.JSONDecodeError as e:
        log.warning(f'Failed to parse patterns file: {e}')
        return None


class BiometricModelsPlugin(ExtractionPlugin):

    def __init__(self):
        super().__init__()
        self._custom_patterns = load_patterns()
        self._extensions = self._build_extensions()
        self._patterns = self._build_patterns()

    def _build_extensions(self):
        ext = {k: v.copy() for k, v in DEFAULT_EXTENSIONS.items()}
        if self._custom_patterns:
            for bt, config in self._custom_patterns.items():
                if 'extensions' in config:
                    ext.setdefault(bt, set()).update(config['extensions'])
        return ext

    def _build_patterns(self):
        pat = {k: v.copy() for k, v in DEFAULT_PATTERNS.items()}
        if self._custom_patterns:
            for bt, config in self._custom_patterns.items():
                if 'patterns' in config:
                    pat.setdefault(bt, []).extend(config['patterns'])
        return pat

    def plugin_info(self):
        plugin_info = PluginInfo(
            id=PluginId('biometric_scanner', 'models', 'BiometricModelsPlugin'),
            version='1.2.0',
            description='Detect pre-computed biometric models and embedding caches',
            author=Author('Biometric Scanner', 'biometric@example.com', 'NFI'),
            maturity=MaturityLevel.PROOF_OF_CONCEPT,
            webpage_url='https://github.com/Lund-University-Hansken-Research-Group/hansken-biometric-scanner',
            matcher=self._build_matcher(),
            license='Apache License 2.0'
        )
        log.info('pluginInfo request')
        log.debug(f'returning plugin info: {plugin_info}')
        return plugin_info

    def _build_matcher(self):
        all_extensions = set()
        for exts in self._extensions.values():
            all_extensions.update(exts)
        matcher_parts = [f'file.extension={ext.lstrip(".")}' for ext in sorted(all_extensions)]
        extension_matcher = ' OR '.join(matcher_parts)
        return f'({extension_matcher}) OR $data.type:raw'

    def process(self, trace, data_context):
        file_name = trace.get('file.name') or trace.get('name') or ''
        file_path = Path(file_name) if file_name else None
        ext = file_path.suffix.lower() if file_path else ''
        name_lower = file_name.lower() if file_name else ''

        log.info(f"processing trace {file_name}")

        if not ext:
            ext = self._detect_pickle_extension(trace)
            if ext:
                log.info(f"Detected pickle file by magic bytes, treating as {ext}")

        detected = self._detect_biometric_model(ext, name_lower, trace, data_context)

        if detected:
            log.info(f"Detected biometric model in {file_name}")
        else:
            log.debug(f"No biometric model detected in {file_name}")

    def _detect_biometric_model(self, ext, name_lower, trace, data_context):
        model_type = self._detect_type(ext, name_lower)
        if not model_type:
            return False

        framework = self._detect_framework(name_lower, ext)
        detected_by = 'extension_pattern'

        if model_type.startswith('unknown') and ext in {'.pkl', '.pickle'}:
            content_type, content_confidence, content_detected_by = self._detect_from_content(trace)
            if content_type:
                model_type = content_type
                confidence = content_confidence
                detected_by = content_detected_by
            else:
                confidence = 'low'
        elif model_type == 'embeddingCache':
            confidence = self._check_embedding_content(trace, data_context)
        elif model_type == 'faceRecognition':
            confidence = 'high'
        elif model_type == 'voiceBiometric':
            confidence = 'medium'
        else:
            confidence = 'low'

        trace.update('biometricModel.type', model_type)
        trace.update('biometricModel.framework', framework)
        trace.update('biometricModel.detectedBy', detected_by)
        trace.update('biometricModel.confidence', confidence)

        self._add_child_details(trace, model_type, framework, ext)

        return True

    def _detect_type(self, ext, name_lower):
        for bt, extensions in self._extensions.items():
            if ext in extensions:
                patterns = self._patterns.get(bt, [])
                if any(p in name_lower for p in patterns):
                    return bt
                return f'unknown{bt}'
        return None

    def _detect_pickle_extension(self, trace):
        try:
            data_stream = trace.open()
            header = data_stream.read(2)
            data_stream.close()
            if len(header) >= 2 and header[0] == 0x80 and 2 <= header[1] <= 5:
                return '.pkl'
        except Exception:
            pass
        return ''

    def _detect_from_content(self, trace):
        try:
            data = self._load_pickle(trace)
        except Exception as e:
            log.debug(f"Failed to load pickle for content analysis: {e}")
            return None, None, None

        detections = []

        if isinstance(data, dict):
            keys = list(data.keys())

            for key in keys:
                key_lower = key.lower()
                if any(kw in key_lower for kw in BIOMETRIC_CONTENT_KEYWORDS):
                    value = data[key]
                    if isinstance(value, (list, tuple)):
                        for item in value[:5]:
                            if isinstance(item, np.ndarray):
                                dim = item.shape[-1] if item.ndim > 0 else 0
                                if dim in FACE_EMBEDDING_DIMS:
                                    detections.append(f"{dim}-dim encoding arrays")
                                    break
                            elif isinstance(item, (list, tuple)) and len(item) in FACE_EMBEDDING_DIMS:
                                detections.append(f"{len(item)}-dim encoding list")
                                break

            if any(kw in str(keys).lower() for kw in NAME_CONTENT_KEYWORDS):
                detections.append("names/labels field")

        elif isinstance(data, (list, tuple)):
            for item in data[:5]:
                if isinstance(item, np.ndarray):
                    dim = item.shape[-1] if item.ndim > 0 else 0
                    if dim in FACE_EMBEDDING_DIMS:
                        detections.append(f"{dim}-dim encoding arrays")
                        break

        if detections:
            model_type = 'faceRecognition' if 'encoding arrays' in ' '.join(detections) or 'encoding list' in ' '.join(detections) else 'embeddingCache'
            return model_type, 'high', 'content_analysis'

        return None, None, None

    def _detect_framework(self, name_lower, ext):
        for fw, patterns in MODEL_FRAMEWORKS.items():
            if any(p in name_lower for p in patterns):
                return fw
        if ext == '.pt' or ext == '.pth':
            return 'pytorch'
        if ext == '.pb':
            return 'tensorflow'
        if ext == '.h5':
            return 'keras'
        if ext == '.dat':
            return 'dlib'
        return 'unknown'

    def _check_embedding_content(self, trace, data_context):
        if self._has_embedding_pattern(trace, data_context):
            return 'high'
        return 'low'

    def _has_embedding_pattern(self, trace, data_context):
        try:
            data = self._load_pickle(trace)
            if isinstance(data, dict):
                for key in data.keys():
                    key_lower = key.lower()
                    for patterns in self._patterns.values():
                        if any(p in key_lower for p in patterns):
                            return True
            elif isinstance(data, (list, tuple)):
                return True
        except Exception:
            pass
        return False

    def _load_pickle(self, trace):
        data_stream = trace.open()
        return Unpickler(data_stream).load()

    def _add_child_details(self, trace, model_type, framework, ext):
        child_builder = trace.child_builder('modelInfo')
        child_builder.update({
            'modelInfo.type': model_type,
            'modelInfo.framework': framework,
            'modelInfo.fileExtension': ext,
        }).build()


if __name__ == '__main__':
    from hansken_extraction_plugin.runtime import run_plugin
    run_plugin(BiometricModelsPlugin)