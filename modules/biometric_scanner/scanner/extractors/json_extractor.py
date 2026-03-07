from pathlib import Path
from typing import Dict, Any, List
import json
import xml.etree.ElementTree as ET

from ..config import BiometricConfig


BIOMETRIC_KEYS = [
    "fingerprint", "face", "facial", "iris", "voice", "gait",
    "biometric", "touch_id", "face_id", "template", "biometric_data",
    "biometric_type", "biometric_id", "encryption", "model", "feature"
]


class JsonExtractor:
    def __init__(self, config: BiometricConfig = None):
        self.config = config or BiometricConfig()
    
    def can_extract(self, filepath: Path) -> bool:
        ext = filepath.suffix.lower()
        if ext in {'.json', '.xml'}:
            return True
        return False
    
    def extract(self, filepath: Path, biometric_type: str) -> Dict[str, Any]:
        ext = filepath.suffix.lower()
        
        if ext == '.json':
            return self._extract_json(filepath)
        elif ext == '.xml':
            return self._extract_xml(filepath)
        
        return {"file_type": "unknown", "error": "Unsupported format"}
    
    def _extract_json(self, filepath: Path) -> Dict[str, Any]:
        result = {
            "file_type": "json",
            "biometric_keys_found": [],
            "biometric_data": {},
            "structure_summary": {}
        }
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            
            result["structure_summary"] = self._summarize_json(data)
            result = self._find_biometric_data(data, result)
            
        except json.JSONDecodeError as e:
            result["error"] = f"JSON parse error: {str(e)}"
        except (IOError, OSError) as e:
            result["error"] = str(e)
        
        return result
    
    def _extract_xml(self, filepath: Path) -> Dict[str, Any]:
        result = {
            "file_type": "xml",
            "biometric_elements_found": [],
            "biometric_data": {},
            "structure_summary": {}
        }
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            result["structure_summary"] = {
                "root_tag": root.tag,
                "total_elements": len(list(root.iter()))
            }
            
            result = self._find_biometric_xml(root, result)
            
        except ET.ParseError as e:
            result["error"] = f"XML parse error: {str(e)}"
        except (IOError, OSError) as e:
            result["error"] = str(e)
        
        return result
    
    def _summarize_json(self, data: Any, depth: int = 0, max_depth: int = 5) -> Dict:
        if depth > max_depth:
            return {"type": "nested", "truncated": True}
        
        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys())[:20],
                "key_count": len(data)
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "item_type": self._summarize_json(data[0], depth + 1) if data else "empty"
            }
        elif isinstance(data, str):
            return {"type": "string", "length": len(data)}
        elif isinstance(data, (int, float)):
            return {"type": "number"}
        elif isinstance(data, bool):
            return {"type": "boolean"}
        elif data is None:
            return {"type": "null"}
        
        return {"type": str(type(data).__name__)}
    
    def _find_biometric_data(self, data: Any, result: Dict, path: str = "") -> Dict:
        if isinstance(data, dict):
            for key, value in data.items():
                key_lower = key.lower()
                
                if any(bio_key in key_lower for bio_key in BIOMETRIC_KEYS):
                    result["biometric_keys_found"].append(f"{path}{key}")
                    
                    if isinstance(value, (dict, list, str, int)):
                        result["biometric_data"][f"{path}{key}"] = self._sanitize_value(value)
                
                if isinstance(value, (dict, list)):
                    result = self._find_biometric_data(value, result, f"{path}{key}.")
        
        elif isinstance(data, list):
            for i, item in enumerate(data[:100]):
                if isinstance(item, (dict, list)):
                    result = self._find_biometric_data(item, result, f"{path}[{i}].")
        
        return result
    
    def _find_biometric_xml(self, root: ET.Element, result: Dict) -> Dict:
        for elem in root.iter():
            tag_lower = elem.tag.lower()
            
            if any(bio_key in tag_lower for bio_key in BIOMETRIC_KEYS):
                result["biometric_elements_found"].append(elem.tag)
                
                if elem.text and elem.text.strip():
                    result["biometric_data"][elem.tag] = self._sanitize_value(elem.text.strip())
                
                for attr, value in elem.attrib.items():
                    if any(bio_key in attr.lower() for bio_key in BIOMETRIC_KEYS):
                        result["biometric_data"][f"{elem.tag}@{attr}"] = self._sanitize_value(value)
        
        return result
    
    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            if len(value) > 500:
                return value[:500] + "..."
            if any(p in value.lower() for p in ["password", "token", "secret", "key"]):
                return "[REDACTED]"
        return value
