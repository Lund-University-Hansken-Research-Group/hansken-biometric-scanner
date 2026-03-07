from dataclasses import dataclass, field
from typing import Dict, List
import json
import os


BIOMETRIC_TYPES = {
    "fingerprint": {
        "patterns": [
            r"fingerprint", r"\.fp\b", r"fpd\b", r"fpdata",
            r"finger", r"biometric", r"fingerprint\.db",
            r"fingerprint\.dat", r"fp_template", r"biometric_auth"
        ],
        "extensions": [".db", ".dat", ".bin", ".sqlite", ".sqlite3"],
        "confidence": "high"
    },
    "facial_recognition": {
        "patterns": [
            r"face", r"facial", r"facelock", r"faceunlock",
            r"face_id", r"faceprint", r"face_data", r"face_template",
            r"face\.db", r"face\.dat", r"facematch"
        ],
        "extensions": [".db", ".dat", ".bin", ".sqlite", ".sqlite3", ".xml"],
        "confidence": "high"
    },
    "iris": {
        "patterns": [
            r"iris", r"irisprint", r"irisc", r"eye_scan",
            r"iris\.db", r"iris\.dat", r"iris_template"
        ],
        "extensions": [".db", ".dat", ".bin"],
        "confidence": "high"
    },
    "voice": {
        "patterns": [
            r"voice", r"voiceprint", r"voice_id", r"voice_template",
            r"voiceauth", r"vmd\b", r"voice\.db", r"speech",
            r"audio_biometric", r"voiceprint"
        ],
        "extensions": [".vmd", ".db", ".dat", ".bin", ".wav", ".mp3"],
        "confidence": "medium"
    },
    "gait": {
        "patterns": [
            r"gait", r"motion", r"activity", r"sensor",
            r"step", r"accelerometer", r"gyroscope",
            r"movement", r"gesture", r"motion\.db"
        ],
        "extensions": [".db", ".csv", ".bin", ".dat", ".json"],
        "confidence": "low"
    },
    "multimodal": {
        "patterns": [
            r"biometric", r"biometrics", r"biometric_auth",
            r"biometric_template", r"biometric_data"
        ],
        "extensions": [".db", ".dat", ".bin", ".sqlite", ".sqlite3"],
        "confidence": "medium"
    }
}


MAGIC_SIGNATURES: Dict[str, bytes] = {
    "sqlite": b"SQLite format 3\x00",
    "android_backup": b"ANDROID BACKUP",
    "zip": b"PK\x03\x04",
    "gzip": b"\x1f\x8b",
}


ANDROID_PATHS = [
    "data/system/",
    "data/system/users/",
    "data/data/com.android.",
    "data/user/",
    "data/user_de/",
    "system/",
    "misc/",
    "vendor/",
]


IOS_PATHS = [
    "Library/",
    "private/var/",
    "SystemPreferences/",
    "BiometricAuthentication/",
    "FaceID/",
    "TouchID/",
]


@dataclass
class BiometricConfig:
    biometric_types: Dict = field(default_factory=lambda: BIOMETRIC_TYPES)
    magic_signatures: Dict = field(default_factory=lambda: MAGIC_SIGNATURES)
    android_paths: List[str] = field(default_factory=lambda: ANDROID_PATHS)
    ios_paths: List[str] = field(default_factory=lambda: IOS_PATHS)
    max_file_size: int = 100 * 1024 * 1024
    binary_read_size: int = 8192
    custom_patterns: Dict = field(default_factory=dict)

    def load_custom_patterns(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.custom_patterns = json.load(f)
                for bt, config in self.custom_patterns.items():
                    if bt in self.biometric_types:
                        self.biometric_types[bt]["patterns"].extend(config.get("patterns", []))
                        self.biometric_types[bt]["extensions"].extend(config.get("extensions", []))
                    else:
                        self.biometric_types[bt] = config


DEFAULT_CONFIG = BiometricConfig()
