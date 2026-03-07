from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Set
import re

from ..config import BiometricConfig


@dataclass
class DetectionResult:
    biometric_type: str
    confidence: str
    detection_method: str
    matched_pattern: Optional[str] = None


BIOMETRIC_CONTENT_PATTERNS = {
    "fingerprint": [
        rb"fingerprint", rb"finger_print", rb"fingerprint_template",
        rb"fp_data", rb"biometric_fingerprint", rb"fingerprint\.db",
        rb"touch_id", rb"fpd_", rb"finger_id"
    ],
    "facial_recognition": [
        rb"face", rb"facial", rb"face_recognition", rb"face_template",
        rb"face_id", rb"face_data", rb"faceprint", rb"facelock",
        rb"face_unlock", rb"biometric_face"
    ],
    "iris": [
        rb"iris", rb"iris_template", rb"irisprint", rb"eye_scan",
        rb"irisc", rb"biometric_iris"
    ],
    "voice": [
        rb"voice", rb"voiceprint", rb"voice_template", rb"voice_id",
        rb"voice_auth", rb"speech_biometric", rb"audio_biometric",
        rb"vmd", rb"voice_model"
    ],
    "gait": [
        rb"gait", rb"motion", rb"activity", rb"step_count",
        rb"accelerometer", rb"gyroscope", rb"movement_pattern",
        rb"gesture", rb"motion_data"
    ],
    "multimodal": [
        rb"biometric", rb"biometric_data", rb"biometric_template",
        rb"biometric_auth", rb"biometric_id"
    ]
}


class ContentDetector:
    def __init__(self, config: BiometricConfig = None):
        self.config = config or BiometricConfig()
        self.compiled_patterns = self._compile_patterns()
        self.scannable_extensions = {
            '.db', '.sqlite', '.sqlite3', '.xml', '.json', 
            '.txt', '.log', '.dat', '.bin'
        }

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        compiled = {}
        for bt, patterns in BIOMETRIC_CONTENT_PATTERNS.items():
            compiled[bt] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled

    def detect(self, filepath: Path) -> List[DetectionResult]:
        results = []
        
        if not self._is_scannable(filepath):
            return results
        
        try:
            content = self._read_file_content(filepath)
        except (IOError, OSError):
            return results

        if not content:
            return results

        content_lower = content.lower()
        
        for bt, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    results.append(DetectionResult(
                        biometric_type=bt,
                        confidence="medium",
                        detection_method="content",
                        matched_pattern=pattern.pattern.decode() if isinstance(pattern.pattern, bytes) else str(pattern.pattern)
                    ))
                    break
        
        return results

    def _is_scannable(self, filepath: Path) -> bool:
        ext = filepath.suffix.lower()
        
        if ext in {'.db', '.sqlite', '.sqlite3', '.xml', '.json', '.txt', '.log', '.dat'}:
            return True
        
        try:
            with open(filepath, 'rb') as f:
                header = f.read(16)
                if header.startswith(b"SQLite format 3\x00"):
                    return True
        except (IOError, OSError):
            pass
        
        return False

    def _read_file_content(self, filepath: Path) -> bytes:
        try:
            file_size = filepath.stat().st_size
            if file_size > self.config.max_file_size:
                read_size = self.config.binary_read_size
            else:
                read_size = min(file_size, 512 * 1024)
            
            with open(filepath, 'rb') as f:
                return f.read(read_size)
        except (IOError, OSError):
            return b""
