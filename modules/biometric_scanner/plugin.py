import csv
import io
import json
import os
import pickletools
import zipfile
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from hansken_extraction_plugin.api.extraction_plugin import ExtractionPlugin
from hansken_extraction_plugin.api.plugin_info import (
    Author,
    MaturityLevel,
    PluginId,
    PluginInfo,
    PluginResources,
)
from logbook import Logger

log = Logger(__name__)

PLUGIN_VERSION = '1.5.0'
MAX_INSPECTION_BYTES = int(os.environ.get('BIOMETRIC_MAX_INSPECTION_BYTES', 64 * 1024 * 1024))
MAX_TEXT_BYTES = min(MAX_INSPECTION_BYTES, 8 * 1024 * 1024)
MAX_CONTAINER_ENTRIES = 256
MAX_JSON_NODES = 10_000

DEFAULT_EXTENSIONS = {
    'faceRecognition': {
        '.pkl', '.pickle', '.dat', '.pb', '.pt', '.pth', '.h5', '.onnx',
        '.caffemodel', '.uff',
    },
    'voiceBiometric': {
        '.pt', '.pth', '.pb', '.h5', '.onnx', '.wav', '.m4a', '.mp3',
        '.pkl', '.pickle',
    },
    'embeddingCache': {'.npy', '.npz', '.pkl', '.pickle', '.json', '.csv'},
}

DEFAULT_PATTERNS = {
    'faceRecognition': [
        'face', 'facial', 'dlib', 'facenet', 'arcface', 'retina', 'vgg',
        'face_embedding', 'face-embedding', 'face_encoding', 'face-encoding',
    ],
    'voiceBiometric': [
        'voice', 'speaker', 'voiceprint', 'xvector', 'x-vector', 'speech',
        'audio_embedding', 'audio-embedding', 'speaker_embedding',
    ],
    'embeddingCache': ['embedding', 'embeddings', 'encoding', 'encodings'],
}

MODEL_FRAMEWORKS = {
    'dlib': ['dlib', 'shape_predictor'],
    'tensorflow': ['tensorflow', 'tf_'],
    'pytorch': ['pytorch', 'torch'],
    'keras': ['keras'],
    'onnx': ['onnx'],
    'scikit': ['sklearn', 'scikit'],
    'caffe': ['caffe', 'caffemodel'],
}

FRAMEWORK_BY_EXTENSION = {
    '.pt': 'pytorch',
    '.pth': 'pytorch',
    '.pb': 'tensorflow',
    '.h5': 'keras',
    '.onnx': 'onnx',
    '.dat': 'dlib',
    '.caffemodel': 'caffe',
}

FACE_EMBEDDING_DIMS = {128, 256, 512, 1024}

FACE_CONTENT_KEYWORDS = {
    'face', 'faces', 'facial', 'face_encodings', 'face_embeddings',
    'facevector', 'facenet', 'dlib_encodings', 'arcface',
}
VOICE_CONTENT_KEYWORDS = {
    'voice', 'speaker', 'voiceprint', 'xvector', 'x-vector', 'speech',
    'audio_embedding', 'speaker_embedding',
}
EMBEDDING_CONTENT_KEYWORDS = {
    'encoding', 'encodings', 'embedding', 'embeddings', 'vectors', 'features',
}
NAME_CONTENT_KEYWORDS = {
    'names', 'labels', 'ids', 'identities', 'persons', 'people', 'subjects',
}

INSPECTABLE_EXTENSIONS = {'.pkl', '.pickle', '.npy', '.npz', '.json', '.csv'}


class InspectionSkipped(Exception):
    pass


def _normalise_extension(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    extension = value.strip().lower()
    if not extension:
        return None
    if not extension.startswith('.'):
        extension = f'.{extension}'
    return extension


def _normalise_patterns(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip().lower() for value in values if str(value).strip()]


def load_patterns() -> Optional[dict[str, Any]]:
    configured_path = os.environ.get('BIOMETRIC_PATTERNS_PATH')
    candidates = []
    if configured_path:
        candidates.append(Path(configured_path))
    candidates.extend([Path(__file__).with_name('patterns.json'), Path('/app/patterns.json')])

    seen = set()
    for patterns_path in candidates:
        resolved = str(patterns_path)
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            with patterns_path.open('r', encoding='utf-8') as handle:
                custom = json.load(handle)
            if not isinstance(custom, dict):
                log.warning(f'Ignoring patterns file {patterns_path}: top-level value must be an object')
                continue
            log.info(f'Loaded custom patterns from {patterns_path}')
            return custom
        except FileNotFoundError:
            continue
        except (OSError, json.JSONDecodeError) as error:
            log.warning(f'Ignoring patterns file {patterns_path}: {error}')

    log.debug('No valid custom patterns file found; using defaults')
    return None


class BiometricModelsPlugin(ExtractionPlugin):

    def __init__(self):
        super().__init__()
        self._custom_patterns = load_patterns()
        self._extensions = self._build_extensions()
        self._patterns = self._build_patterns()

    def _build_extensions(self) -> dict[str, set[str]]:
        extensions = {key: value.copy() for key, value in DEFAULT_EXTENSIONS.items()}
        if not self._custom_patterns:
            return extensions

        for biometric_type, config in self._custom_patterns.items():
            if not isinstance(config, dict):
                log.warning(f'Ignoring invalid configuration for {biometric_type}')
                continue
            configured = config.get('extensions', [])
            if not isinstance(configured, list):
                log.warning(f'Ignoring extensions for {biometric_type}: expected a list')
                continue
            normalised = {
                extension
                for value in configured
                if (extension := _normalise_extension(value)) is not None
            }
            extensions.setdefault(str(biometric_type), set()).update(normalised)
        return extensions

    def _build_patterns(self) -> dict[str, list[str]]:
        patterns = {key: value.copy() for key, value in DEFAULT_PATTERNS.items()}
        if not self._custom_patterns:
            return patterns

        for biometric_type, config in self._custom_patterns.items():
            if not isinstance(config, dict):
                continue
            configured = _normalise_patterns(config.get('patterns', []))
            target = patterns.setdefault(str(biometric_type), [])
            target.extend(pattern for pattern in configured if pattern not in target)
        return patterns

    def plugin_info(self) -> PluginInfo:
        plugin_info = PluginInfo(
            id=PluginId(
                domain='biometric_scanner',
                category='models',
                name='BiometricModelsPlugin',
            ),
            version=PLUGIN_VERSION,
            description='Detect pre-computed biometric models and embedding caches',
            author=Author(
                'Lund University Hansken Research Group',
                'dylan.pashley@svet.lu.se',
                'LU',
            ),
            maturity=MaturityLevel.PROOF_OF_CONCEPT,
            webpage_url=(
                'https://github.com/Lund-University-Hansken-Research-Group/'
                'hansken-biometric-scanner'
            ),
            matcher=self._build_matcher(),
            license='GNU General Public License v3.0',
            resources=PluginResources(
                maximum_cpu=1,
                maximum_memory=512,
                maximum_workers=2,
            ),
        )
        log.info('pluginInfo request')
        log.debug(f'returning plugin info: {plugin_info}')
        return plugin_info

    def _build_matcher(self) -> str:
        all_extensions = sorted({extension for values in self._extensions.values() for extension in values})
        extension_matcher = ' OR '.join(
            f'file.extension={extension.lstrip(".")}' for extension in all_extensions
        )
        filename_matcher = ' OR '.join(
            f'file.name=*{extension}' for extension in all_extensions
        )
        candidate_matcher = f'({extension_matcher}) OR ({filename_matcher})'
        return f'file.misc.biometricModelType!=* AND ({candidate_matcher})'

    def process(self, trace, data_context) -> None:
        file_name = trace.get('file.name') or trace.get('name') or ''
        name_lower = str(file_name).lower()

        extension = _normalise_extension(trace.get('file.extension'))
        if extension is None and file_name:
            extension = Path(str(file_name)).suffix.lower()
        extension = extension or ''

        log.info(f'processing trace {file_name}')
        detected = self._detect_biometric_model(
            extension,
            name_lower,
            trace,
            data_context,
        )

        if detected:
            log.info(f'Detected biometric model in {file_name}')
        else:
            log.debug(f'No biometric model detected in {file_name}')

    def _detect_biometric_model(self, extension, name_lower, trace, data_context) -> bool:
        model_type, matched_patterns = self._detect_type(extension, name_lower)
        detected_by = 'extension_and_name'
        confidence = None

        content_result = None
        if extension in INSPECTABLE_EXTENSIONS:
            content_result = self._detect_from_content(extension, trace, data_context)

        if content_result:
            content_type, content_confidence, content_detected_by = content_result
            if model_type is None or model_type == 'embeddingCache' or content_confidence == 'high':
                model_type = content_type
                confidence = content_confidence
                detected_by = content_detected_by

        if model_type is None:
            return False

        if confidence is None:
            if model_type == 'faceRecognition':
                confidence = 'high' if matched_patterns else 'low'
            elif model_type == 'voiceBiometric':
                confidence = 'medium' if matched_patterns else 'low'
            else:
                confidence = 'medium' if matched_patterns else 'low'

        framework = self._detect_framework(name_lower, extension)
        trace.update('file.misc.biometricModelType', model_type)
        trace.update('file.misc.biometricModelFramework', framework)
        trace.update('file.misc.biometricModelDetectedBy', detected_by)
        trace.update('file.misc.biometricModelConfidence', confidence)
        return True

    def _detect_type(self, extension: str, name_lower: str) -> tuple[Optional[str], list[str]]:
        candidates = []
        for biometric_type, extensions in self._extensions.items():
            if extension not in extensions:
                continue
            matched = [
                pattern
                for pattern in self._patterns.get(biometric_type, [])
                if pattern in name_lower
            ]
            if matched:
                score = (len(matched), sum(map(len, matched)), max(map(len, matched)))
                candidates.append((score, biometric_type, matched))

        if not candidates:
            return None, []

        candidates.sort(reverse=True)
        _, biometric_type, matched = candidates[0]
        return biometric_type, matched

    def _detect_from_content(self, extension, trace, data_context):
        try:
            if extension in {'.pkl', '.pickle'}:
                return self._inspect_pickle(trace, data_context)
            if extension == '.npy':
                return self._inspect_npy(trace, data_context)
            if extension == '.npz':
                return self._inspect_npz(trace, data_context)
            if extension == '.json':
                return self._inspect_json(trace, data_context)
            if extension == '.csv':
                return self._inspect_csv(trace, data_context)
        except InspectionSkipped as error:
            log.debug(f'Content inspection skipped: {error}')
        except Exception as error:
            log.debug(f'Content inspection failed for {extension}: {error}')
        return None

    def _read_stream(self, trace, data_context, limit: int) -> bytes:
        data_size = getattr(data_context, 'data_size', None)
        if isinstance(data_size, int) and data_size > limit:
            raise InspectionSkipped(f'data stream is {data_size} bytes; limit is {limit}')

        data_type = getattr(data_context, 'data_type', None)
        try:
            if data_type:
                stream = trace.open(data_type=data_type)
            else:
                stream = trace.open()
        except Exception:
            stream = trace.open()

        with stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise InspectionSkipped(f'data stream exceeds {limit} bytes')
        return data

    def _inspect_pickle(self, trace, data_context):
        data = self._read_stream(trace, data_context, MAX_INSPECTION_BYTES)
        strings = []
        integers = set()

        for opcode, argument, _position in pickletools.genops(data):
            if isinstance(argument, str):
                strings.append(argument.lower())
            elif isinstance(argument, bytes):
                strings.append(argument.decode('utf-8', errors='ignore').lower())
            elif isinstance(argument, int):
                integers.add(argument)

        return self._classify_content(strings, integers, 'pickle_opcode_analysis')

    def _read_npy_header(self, stream) -> tuple[tuple[int, ...], Any]:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in {(2, 0), (3, 0)}:
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f'Unsupported NPY version: {version}')
        return tuple(shape), dtype

    def _inspect_npy(self, trace, data_context):
        data = self._read_stream(trace, data_context, MAX_INSPECTION_BYTES)
        shape, dtype = self._read_npy_header(io.BytesIO(data))
        if dtype.hasobject:
            return None
        if shape and shape[-1] in FACE_EMBEDDING_DIMS:
            return 'embeddingCache', 'high', 'npy_header_analysis'
        return None

    def _inspect_npz(self, trace, data_context):
        data = self._read_stream(trace, data_context, MAX_INSPECTION_BYTES)
        strings = []
        dimensions = set()

        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_CONTAINER_ENTRIES:
                raise InspectionSkipped(f'NPZ has more than {MAX_CONTAINER_ENTRIES} entries')
            for entry in entries:
                strings.append(entry.filename.lower())
                if not entry.filename.lower().endswith('.npy'):
                    continue
                with archive.open(entry) as stream:
                    shape, dtype = self._read_npy_header(stream)
                if not dtype.hasobject and shape:
                    dimensions.add(shape[-1])

        return self._classify_content(strings, dimensions, 'npz_header_analysis')

    def _inspect_json(self, trace, data_context):
        data = self._read_stream(trace, data_context, MAX_TEXT_BYTES)
        payload = json.loads(data.decode('utf-8-sig'))
        strings, dimensions = self._summarise_json(payload)
        return self._classify_content(strings, dimensions, 'json_structure_analysis')

    def _summarise_json(self, payload) -> tuple[list[str], set[int]]:
        strings = []
        dimensions = set()
        stack = [payload]
        visited = 0

        while stack and visited < MAX_JSON_NODES:
            value = stack.pop()
            visited += 1
            if isinstance(value, dict):
                for key, item in value.items():
                    strings.append(str(key).lower())
                    stack.append(item)
            elif isinstance(value, list):
                dimensions.add(len(value))
                stack.extend(value[:64])
            elif isinstance(value, str):
                strings.append(value.lower())

        return strings, dimensions

    def _inspect_csv(self, trace, data_context):
        data = self._read_stream(trace, data_context, MAX_TEXT_BYTES)
        text = data.decode('utf-8-sig', errors='replace')
        sample = text[:65536]
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel

        rows = csv.reader(io.StringIO(text), dialect)
        strings = []
        dimensions = set()
        for row_number, row in enumerate(rows):
            if row_number >= 64:
                break
            dimensions.add(len(row))
            if row_number == 0:
                strings.extend(cell.lower() for cell in row)

        return self._classify_content(strings, dimensions, 'csv_structure_analysis')

    def _classify_content(self, strings: Iterable[str], dimensions: Iterable[int], detected_by: str):
        combined = ' '.join(strings).lower()
        dimensions = set(dimensions)
        has_embedding_dimension = bool(dimensions & FACE_EMBEDDING_DIMS)
        has_names = any(keyword in combined for keyword in NAME_CONTENT_KEYWORDS)
        has_face = any(keyword in combined for keyword in FACE_CONTENT_KEYWORDS)
        has_voice = any(keyword in combined for keyword in VOICE_CONTENT_KEYWORDS)
        has_embedding = any(keyword in combined for keyword in EMBEDDING_CONTENT_KEYWORDS)

        if has_voice and (has_embedding or has_embedding_dimension):
            return 'voiceBiometric', 'high', detected_by
        if has_face and (has_embedding or has_embedding_dimension or has_names):
            return 'faceRecognition', 'high', detected_by
        if has_embedding_dimension and (has_embedding or has_names):
            return 'embeddingCache', 'high', detected_by
        if has_embedding and has_names:
            return 'embeddingCache', 'medium', detected_by
        return None

    def _detect_framework(self, name_lower: str, extension: str) -> str:
        for framework, patterns in MODEL_FRAMEWORKS.items():
            if any(pattern in name_lower for pattern in patterns):
                return framework
        return FRAMEWORK_BY_EXTENSION.get(extension, 'unknown')


if __name__ == '__main__':
    from hansken_extraction_plugin.runtime import run_plugin
    run_plugin(BiometricModelsPlugin)
