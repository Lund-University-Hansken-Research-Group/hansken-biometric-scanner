from pathlib import Path

import numpy as np
from pickle import Unpickler

from hansken_extraction_plugin.api.extraction_plugin import ExtractionPlugin
from hansken_extraction_plugin.api.plugin_info import Author, MaturityLevel, PluginId, PluginInfo
from logbook import Logger

log = Logger(__name__)


FACE_MODEL_EXTENSIONS = {
    '.pkl', '.pickle',
    '.dat',  # dlib shape_predictor
    '.pb',   # TensorFlow protobuf
    '.pt', '.pth',  # PyTorch
    '.h5',   # Keras
    '.caffemodel',  # Caffe
    '.uff',  # TensorRT
}

AUDIO_MODEL_EXTENSIONS = {
    '.pt', '.pth', '.pb', '.h5',
    '.wav', '.m4a', '.mp3',
    '.pkl', '.pickle',
}

EMBEDDING_CACHE_EXTENSIONS = {
    '.npy', '.npz',
    '.pkl', '.pickle',
    '.json',
    '.csv',
}

TRAINING_DATA_PATTERNS = [
    'embedding', 'embeddings', 'encoding', 'encodings',
    'face', 'faces', 'facial',
    'voice', 'speaker', 'voiceprint',
    'dlib', 'facenet', 'arcface', 'retinaface',
    'resnet', 'vggface', 'facenet',
    'xvector', 'x-vector', 'speaker_embedding',
    'verification', 'identification',
]

MODEL_FRAMEWORKS = {
    'dlib': ['dlib', 'shape_predictor'],
    'tensorflow': ['tensorflow', 'tf_', '.pb'],
    'pytorch': ['pytorch', 'torch', '.pt', '.pth'],
    'keras': ['keras', '.h5'],
    'onnx': ['onnx', '.onnx'],
    'scikit': ['sklearn', 'scikit'],
}


class BiometricModelsPlugin(ExtractionPlugin):

    def plugin_info(self):
        plugin_info = PluginInfo(
            id=PluginId('biometric_scanner', 'models', 'BiometricModelsPlugin'),
            version='1.0.0',
            description='Detect pre-computed biometric models and embedding caches',
            author=Author('Biometric Scanner', 'biometric@example.com', 'NFI'),
            maturity=MaturityLevel.PROOF_OF_CONCEPT,
            webpage_url='https://github.com/dp/hansken-biometric-scanner',
            matcher=(
                'file.extension=pkl OR file.extension=pickle OR '
                'file.extension=dat OR '
                'file.extension=pb OR '
                'file.extension=pt OR file.extension=pth OR '
                'file.extension=h5 OR '
                'file.extension=npy OR file.extension=npz OR '
                'file.extension=json OR '
                'file.extension=wav OR file.extension=m4a OR file.extension=mp3'
            ),
            license='Apache License 2.0'
        )
        log.info('pluginInfo request')
        log.debug(f'returning plugin info: {plugin_info}')
        return plugin_info

    def process(self, trace, data_context):
        file_name = trace.get('file.name')
        file_path = Path(file_name) if file_name else None
        ext = file_path.suffix.lower() if file_path else ''
        name_lower = file_name.lower() if file_name else ''

        log.info(f"processing trace {file_name}")

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

        trace.update('biometricModel.type', model_type)
        trace.update('biometricModel.framework', framework)
        trace.update('biometricModel.detectedBy', 'extension_pattern')

        if model_type == 'faceRecognition':
            confidence = 'high'
        elif model_type == 'voiceBiometric':
            confidence = 'medium'
        elif model_type == 'embeddingCache':
            confidence = self._check_embedding_content(trace, data_context)
        else:
            confidence = 'low'

        trace.update('biometricModel.confidence', confidence)

        self._add_child_details(trace, model_type, framework, ext)

        return True

    def _detect_type(self, ext, name_lower):
        if ext in FACE_MODEL_EXTENSIONS:
            if any(p in name_lower for p in ['face', 'facial', 'dlib', 'facenet', 'arcface', 'retina', 'vgg']):
                return 'faceRecognition'
            return 'unknownBiometricModel'

        if ext in AUDIO_MODEL_EXTENSIONS:
            if any(p in name_lower for p in ['voice', 'speaker', 'voiceprint', 'xvector', 'speech']):
                return 'voiceBiometric'
            return 'unknownAudioModel'

        if ext in EMBEDDING_CACHE_EXTENSIONS:
            if any(p in name_lower for p in TRAINING_DATA_PATTERNS):
                return 'embeddingCache'

        if ext in {'.pkl', '.pickle'}:
            return self._check_pickle_content(name_lower)

        return None

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

    def _check_pickle_content(self, name_lower):
        if any(p in name_lower for p in TRAINING_DATA_PATTERNS):
            return 'embeddingCache'
        return 'unknownPickle'

    def _check_embedding_content(self, trace, data_context):
        if self._has_embedding_pattern(trace, data_context):
            return 'high'
        return 'low'

    def _has_embedding_pattern(self, trace, data_context):
        try:
            data = self._load_pickle(trace, data_context)
            if isinstance(data, dict):
                for key in data.keys():
                    key_lower = key.lower()
                    if any(p in key_lower for p in TRAINING_DATA_PATTERNS):
                        return True
            elif isinstance(data, (list, tuple)):
                return True
        except Exception:
            pass
        return False

    def _load_pickle(self, trace, data_context):
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
