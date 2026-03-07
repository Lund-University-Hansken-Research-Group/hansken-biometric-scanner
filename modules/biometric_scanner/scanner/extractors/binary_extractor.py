from pathlib import Path
from typing import Dict, Any, List
import re

from ..config import BiometricConfig


BIOMETRIC_BINARY_SIGNATURES = [
    (b"\x00\x00\x00\x00", "padding/null"),
    (b"\x01\x00\x00\x00", "type_1"),
    (b"\x02\x00\x00\x00", "type_2"),
]

BIOMETRIC_STRING_PATTERNS = [
    rb"biometric",
    rb"fingerprint",
    rb"fingerprint_template",
    rb"face[_\-]?data",
    rb"face[_\-]?template",
    rb"iris[_\-]?template",
    rb"voice[_\-]?model",
    rb"touch[_\-]?id",
    rb"face[_\-]?id",
    rb"AES",
    rb"RSA",
    rb"encrypted",
]


class BinaryExtractor:
    def __init__(self, config: BiometricConfig = None):
        self.config = config or BiometricConfig()
    
    def can_extract(self, filepath: Path) -> bool:
        if filepath.stat().st_size == 0:
            return False
        return True
    
    def extract(self, filepath: Path, biometric_type: str) -> Dict[str, Any]:
        result = {
            "file_type": "binary",
            "size_bytes": filepath.stat().st_size,
            "strings_found": [],
            "potential_templates": [],
            "encryption_indicators": [],
            "entropy": None
        }
        
        try:
            with open(filepath, 'rb') as f:
                data = f.read(min(filepath.stat().st_size, 1024 * 1024))
            
            result["strings_found"] = self._extract_strings(data)
            result["potential_templates"] = self._find_template_indicators(data)
            result["encryption_indicators"] = self._detect_encryption(data)
            result["entropy"] = self._calculate_entropy(data)
            
        except (IOError, OSError) as e:
            result["error"] = str(e)
        
        return result
    
    def _extract_strings(self, data: bytes, min_length: int = 4) -> List[str]:
        strings = []
        current = []
        
        for byte in data:
            if 32 <= byte <= 126:
                current.append(chr(byte))
            else:
                if len(current) >= min_length:
                    strings.append(''.join(current))
                current = []
        
        if len(current) >= min_length:
            strings.append(''.join(current))
        
        relevant = []
        for s in strings:
            if any(p in s.lower() for p in ["biometric", "fingerprint", "face", "iris", "voice", "template", "model", "encrypt"]):
                relevant.append(s)
        
        return relevant[:50]
    
    def _find_template_indicators(self, data: bytes) -> List[Dict[str, Any]]:
        indicators = []
        
        for pattern in BIOMETRIC_STRING_PATTERNS:
            matches = re.finditer(re.escape(pattern), data, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 20)
                end = min(len(data), match.end() + 20)
                context = data[start:end]
                
                indicators.append({
                    "pattern": pattern.decode('utf-8', errors='ignore'),
                    "position": match.start(),
                    "context": context.hex()[:40]
                })
        
        return indicators[:20]
    
    def _detect_encryption(self, data: bytes) -> List[str]:
        indicators = []
        
        encryption_strings = [b"AES", b"RSA", b"encrypted", b"cipher", b"DECRYPT", b"ENCRYPT"]
        
        for enc in encryption_strings:
            if enc in data:
                indicators.append(enc.decode('utf-8', errors='ignore'))
        
        if len(data) > 256:
            sample = data[:256]
            unique_bytes = len(set(sample))
            if unique_bytes > 200:
                indicators.append("high_entropy (possible encryption)")
        
        return indicators
    
    def _calculate_entropy(self, data: bytes) -> float:
        if not data:
            return 0.0
        
        from math import log2
        frequencies = {}
        for byte in data:
            frequencies[byte] = frequencies.get(byte, 0) + 1
        
        entropy = 0.0
        for count in frequencies.values():
            probability = count / len(data)
            entropy -= probability * log2(probability)
        
        return round(entropy, 2)
