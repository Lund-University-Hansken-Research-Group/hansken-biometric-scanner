from pathlib import Path
from typing import Dict, List, Any, Optional
import sqlite3
import json

from ..config import BiometricConfig


BIOMETRIC_TABLE_PATTERNS = [
    "fingerprint", "face", "facial", "iris", "voice", "gait", 
    "biometric", "touch", "face_id", "touch_id", "sensor"
]

BIOMETRIC_COLUMN_PATTERNS = [
    "template", "biometric", "fingerprint", "face", "iris", "voice",
    "data", "hash", "encoding", "feature", "model", "signature"
]


class SqliteExtractor:
    def __init__(self, config: BiometricConfig = None):
        self.config = config or BiometricConfig()
    
    def can_extract(self, filepath: Path) -> bool:
        try:
            with open(filepath, 'rb') as f:
                header = f.read(16)
            return header.startswith(b"SQLite format 3\x00")
        except (IOError, OSError):
            return False
    
    def extract(self, filepath: Path, biometric_type: str) -> Dict[str, Any]:
        result = {
            "file_type": "sqlite",
            "tables_found": [],
            "biometric_tables": [],
            "biometric_columns": [],
            "record_counts": {},
            "sample_data": [],
            "encryption": self._detect_encryption(filepath)
        }
        
        try:
            conn = sqlite3.connect(str(filepath))
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0].lower()
                result["tables_found"].append(table[0])
                
                if any(pattern in table_name for pattern in BIOMETRIC_TABLE_PATTERNS):
                    result["biometric_tables"].append(table[0])
                    result = self._extract_table_data(cursor, table[0], result)
                
                for col_pattern in BIOMETRIC_COLUMN_PATTERNS:
                    cursor.execute(f"PRAGMA table_info({table[0]});")
                    columns = cursor.fetchall()
                    for col in columns:
                        if col_pattern in col[1].lower():
                            result["biometric_columns"].append(f"{table[0]}.{col[1]}")
            
            conn.close()
        except sqlite3.Error as e:
            result["error"] = str(e)
        
        return result
    
    def _extract_table_data(self, cursor, table_name: str, result: Dict) -> Dict:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            result["record_counts"][table_name] = count
            
            if count > 0 and count <= 100:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 10;")
                rows = cursor.fetchall()
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = [col[1] for col in cursor.fetchall()]
                
                result["sample_data"].append({
                    "table": table_name,
                    "columns": columns,
                    "rows": [list(row) for row in rows]
                })
        except sqlite3.Error:
            pass
        
        return result
    
    def _detect_encryption(self, filepath: Path) -> Optional[str]:
        try:
            with open(filepath, 'rb') as f:
                header = f.read(112)
            
            if b"SQLite format 3" in header:
                page_size = int.from_bytes(header[16:18], 'big')
                if page_size == 0:
                    return None
                
                text_encoding = header[56:60]
                if text_encoding == b"\x01\x00\x00\x00":
                    return "UTF-8"
                elif text_encoding == b"\x02\x00\x00\x00":
                    return "UTF-16le"
                elif text_encoding == b"\x03\x00\x00\x00":
                    return "UTF-16be"
            
            return None
        except (IOError, OSError):
            return None
