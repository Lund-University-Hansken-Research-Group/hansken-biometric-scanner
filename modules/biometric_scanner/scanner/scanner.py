from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .config import BiometricConfig
from .discoverer import FileDiscoverer
from .detectors import FilenameDetector, MagicDetector, ContentDetector
from .extractors import SqliteExtractor, BinaryExtractor, JsonExtractor


class BiometricScanner:
    def __init__(
        self,
        config: BiometricConfig = None,
        parallel_workers: int = 4,
        extract_data: bool = True
    ):
        self.config = config or BiometricConfig()
        self.parallel_workers = parallel_workers
        self.extract_data = extract_data
        
        self.filename_detector = FilenameDetector(self.config)
        self.magic_detector = MagicDetector(self.config)
        self.content_detector = ContentDetector(self.config)
        
        self.sqlite_extractor = SqliteExtractor(self.config)
        self.binary_extractor = BinaryExtractor(self.config)
        self.json_extractor = JsonExtractor(self.config)
    
    def scan(self, base_path: str, recursive: bool = False) -> Dict[str, Any]:
        discoverer = FileDiscoverer(base_path, recursive, self.config)
        
        files = list(discoverer.discover())
        files_scanned = len([f for f in files if not f.is_directory])
        
        results = []
        
        if self.parallel_workers > 1:
            results = self._scan_parallel(files)
        else:
            results = self._scan_sequential(files)
        
        matches = [r for r in results if r is not None and r.get("biometric_type")]
        
        return {
            "scan_info": {
                "path": base_path,
                "recursive": recursive,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "files_scanned": files_scanned,
                "files_matched": len(matches)
            },
            "results": matches
        }
    
    def _scan_sequential(self, files: List) -> List[Dict]:
        results = []
        for file_info in files:
            if file_info.is_directory:
                continue
            result = self._scan_file(file_info)
            if result:
                results.append(result)
        return results
    
    def _scan_parallel(self, files: List) -> List[Dict]:
        results = []
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {
                executor.submit(self._scan_file, f): f 
                for f in files if not f.is_directory
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception:
                    pass
        
        return results
    
    def _scan_file(self, file_info) -> Optional[Dict]:
        filepath = file_info.path
        detection_methods = set()
        biometric_types = {}
        
        filename_results = self.filename_detector.detect(filepath)
        for r in filename_results:
            detection_methods.add(r.detection_method)
            biometric_types[r.biometric_type] = max(
                biometric_types.get(r.biometric_type, 0),
                {"high": 3, "medium": 2, "low": 1}.get(r.confidence, 1)
            )
        
        magic_results = self.magic_detector.detect(filepath)
        for r in magic_results:
            detection_methods.add(r.detection_method)
            if r.biometric_type not in biometric_types:
                biometric_types[r.biometric_type] = 1
        
        content_results = self.content_detector.detect(filepath)
        for r in content_results:
            detection_methods.add(r.detection_method)
            biometric_types[r.biometric_type] = max(
                biometric_types.get(r.biometric_type, 0),
                {"high": 3, "medium": 2, "low": 1}.get(r.confidence, 1)
            )
        
        if not biometric_types:
            return None
        
        primary_type = max(biometric_types, key=biometric_types.get)
        confidence = "high" if biometric_types[primary_type] >= 3 else \
                    "medium" if biometric_types[primary_type] >= 2 else "low"
        
        result = {
            "file": str(filepath),
            "relative_path": file_info.relative_path,
            "biometric_type": primary_type,
            "confidence": confidence,
            "detection_methods": list(detection_methods),
            "size_bytes": file_info.size_bytes,
            "extracted_data": {}
        }
        
        if self.extract_data:
            result["extracted_data"] = self._extract_file_data(filepath, primary_type)
        
        return result
    
    def _extract_file_data(self, filepath: Path, biometric_type: str) -> Dict[str, Any]:
        extracted = {}
        
        if self.sqlite_extractor.can_extract(filepath):
            extracted = self.sqlite_extractor.extract(filepath, biometric_type)
        elif self.json_extractor.can_extract(filepath):
            extracted = self.json_extractor.extract(filepath, biometric_type)
        elif self.binary_extractor.can_extract(filepath):
            extracted = self.binary_extractor.extract(filepath, biometric_type)
        
        return extracted
