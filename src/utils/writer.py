import os
from pathlib import Path
from abc import ABC, abstractmethod
import csv

class Writer(ABC):
    """"""

    @abstractmethod
    def write(self, data: list[tuple] | tuple):
        """"""
    @abstractmethod
    def close(self):
        """"""
    
    @staticmethod
    def create(output_path: str):
        path = Path(output_path)
        if not path.is_absolute():
            raise ValueError(f"Path must be absolute. Got: {output_path}")
        resolved_path = path.resolve()

        if resolved_path.suffix.lower() == '.csv':
            return CsvWriter(resolved_path)
        raise ValueError(f"Unsupported format: {resolved_path.suffix}")

class CsvWriter(Writer):

    def __init__(self, output_path):
        self.output_path = output_path
        self.file = None
        self.writer = None

    def write(self, data: list[tuple] | tuple):
        """"""
        try:
            if not self.file:
                self.file = open(self.output_path, "w", newline="", encoding="utf-8")
                self.writer = csv.writer(self.file)
            self.writer.writerow(data)
        except OSError as e:
            raise OSError(f"File system error accessing {self.output_path}: {e}") from e

    def clear(self):
        """"""
        self.file.truncate(0)

    def close(self):
        """"""
        if self.file:
            self.file.close()
