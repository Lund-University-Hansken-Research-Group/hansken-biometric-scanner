from pathlib import Path
from typing import List, Generator
from dataclasses import dataclass
import os

from .config import BiometricConfig


@dataclass
class FileInfo:
    path: Path
    relative_path: str
    size_bytes: int
    is_directory: bool


class FileDiscoverer:
    def __init__(self, base_path: str, recursive: bool = False, config: BiometricConfig = None):
        self.base_path = Path(base_path)
        self.recursive = recursive
        self.config = config or BiometricConfig()
        
        if not self.base_path.exists():
            raise ValueError(f"Path does not exist: {base_path}")
    
    def discover(self) -> Generator[FileInfo, None, None]:
        if self.base_path.is_file():
            yield FileInfo(
                path=self.base_path,
                relative_path=self.base_path.name,
                size_bytes=self.base_path.stat().st_size,
                is_directory=False
            )
            return
        
        if self.recursive:
            yield from self._discover_recursive()
        else:
            yield from self._discover_top_level()
    
    def _discover_top_level(self) -> Generator[FileInfo, None, None]:
        try:
            for entry in os.scandir(self.base_path):
                path = Path(entry.path)
                yield FileInfo(
                    path=path,
                    relative_path=path.name,
                    size_bytes=entry.stat().st_size if entry.is_file() else 0,
                    is_directory=entry.is_dir()
                )
        except (IOError, OSError) as e:
            pass
    
    def _discover_recursive(self) -> Generator[FileInfo, None, None]:
        try:
            for root, dirs, files in os.walk(self.base_path):
                root_path = Path(root)
                
                for dirname in dirs:
                    dir_path = root_path / dirname
                    try:
                        yield FileInfo(
                            path=dir_path,
                            relative_path=str(dir_path.relative_to(self.base_path)),
                            size_bytes=0,
                            is_directory=True
                        )
                    except ValueError:
                        pass
                
                for filename in files:
                    file_path = root_path / filename
                    try:
                        stat = file_path.stat()
                        yield FileInfo(
                            path=file_path,
                            relative_path=str(file_path.relative_to(self.base_path)),
                            size_bytes=stat.st_size,
                            is_directory=False
                        )
                    except (IOError, OSError, ValueError):
                        pass
        except (IOError, OSError):
            pass
