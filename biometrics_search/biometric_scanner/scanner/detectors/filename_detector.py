from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import re

from ..config import BiometricConfig


@dataclass
class DetectionResult:
    biometric_type: str
    confidence: str
    detection_method: str
    matched_pattern: Optional[str] = None


class FilenameDetector:
    def __init__(self, config: BiometricConfig = None):
        self.config = config or BiometricConfig()
        self._compile_patterns()

    def _compile_patterns(self):
        self.compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for bt, conf in self.config.biometric_types.items():
            self.compiled_patterns[bt] = [
                re.compile(p, re.IGNORECASE) for p in conf.get("patterns", [])
            ]

    def detect(self, filepath: Path) -> List[DetectionResult]:
        results = []
        filename = filepath.name.lower()
        
        for bt, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(filename):
                    results.append(DetectionResult(
                        biometric_type=bt,
                        confidence=self.config.biometric_types[bt].get("confidence", "medium"),
                        detection_method="filename",
                        matched_pattern=pattern.pattern
                    ))
                    break
        
        return results

    def is_relevant_extension(self, filepath: Path) -> bool:
        ext = filepath.suffix.lower()
        for conf in self.config.biometric_types.values():
            if ext in conf.get("extensions", []):
                return True
        return False
