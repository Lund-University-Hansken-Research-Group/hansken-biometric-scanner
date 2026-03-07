from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from datetime import datetime
import json


@dataclass
class ScanInfo:
    path: str
    recursive: bool
    timestamp: str
    files_scanned: int
    files_matched: int


@dataclass
class BiometricMatch:
    file: str
    relative_path: str
    biometric_type: str
    confidence: str
    detection_methods: List[str]
    size_bytes: int
    extracted_data: Dict[str, Any]


class JsonFormatter:
    def __init__(self, pretty: bool = True):
        self.pretty = pretty
    
    def format(self, scan_info: ScanInfo, results: List[BiometricMatch]) -> str:
        output = {
            "scan_info": asdict(scan_info),
            "results": [asdict(r) for r in results],
            "summary": {
                "total_files_scanned": scan_info.files_scanned,
                "total_biometric_files": scan_info.files_matched,
                "by_type": self._count_by_type(results),
                "by_confidence": self._count_by_confidence(results),
                "by_detection_method": self._count_by_method(results)
            }
        }
        
        if self.pretty:
            return json.dumps(output, indent=2, default=str)
        return json.dumps(output, default=str)
    
    def _count_by_type(self, results: List[BiometricMatch]) -> Dict[str, int]:
        counts = {}
        for r in results:
            counts[r.biometric_type] = counts.get(r.biometric_type, 0) + 1
        return counts
    
    def _count_by_confidence(self, results: List[BiometricMatch]) -> Dict[str, int]:
        counts = {}
        for r in results:
            counts[r.confidence] = counts.get(r.confidence, 0) + 1
        return counts
    
    def _count_by_method(self, results: List[BiometricMatch]) -> Dict[str, int]:
        counts = {}
        for r in results:
            for method in r.detection_methods:
                counts[method] = counts.get(method, 0) + 1
        return counts
    
    def format_file_list(self, results: List[BiometricMatch]) -> str:
        lines = []
        for r in results:
            lines.append(f"{r.relative_path} ({r.biometric_type}, {r.confidence})")
        return "\n".join(lines)
