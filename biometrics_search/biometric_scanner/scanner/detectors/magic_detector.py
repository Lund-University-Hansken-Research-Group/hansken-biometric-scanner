from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

from ..config import BiometricConfig


@dataclass
class DetectionResult:
    biometric_type: str
    confidence: str
    detection_method: str
    matched_pattern: Optional[str] = None


class MagicDetector:
    def __init__(self, config: BiometricConfig = None):
        self.config = config or BiometricConfig()
        self.magic_sigs = self.config.magic_signatures

    def detect(self, filepath: Path) -> List[DetectionResult]:
        results = []
        
        try:
            with open(filepath, 'rb') as f:
                header = f.read(self.config.binary_read_size)
        except (IOError, OSError):
            return results

        for sig_name, sig_bytes in self.magic_sigs.items():
            if header.startswith(sig_bytes):
                results.append(DetectionResult(
                    biometric_type="database",
                    confidence="medium",
                    detection_method="magic_bytes",
                    matched_pattern=sig_name
                ))
        
        if self._detect_sqlite_format(header):
            results.append(DetectionResult(
                biometric_type="database",
                confidence="high",
                detection_method="magic_bytes",
                matched_pattern="sqlite"
            ))
        
        return results

    def _detect_sqlite_format(self, header: bytes) -> bool:
        if len(header) < 16:
            return False
        return header[:16] == b"SQLite format 3\x00"

    def is_relevant_file(self, filepath: Path) -> bool:
        try:
            with open(filepath, 'rb') as f:
                header = f.read(16)
            return any(
                header.startswith(sig) 
                for sig in self.magic_sigs.values()
            ) or self._detect_sqlite_format(header)
        except (IOError, OSError):
            return False
