"""
Data File Writer Module

Provides abstract interfaces and concrete implementations for writing structured 
data to different file formats.

Classes:
    :Writer: Abstract base class defining writer interface
    :CsvWriter: Concrete implementation for CSV file format

Dependencies:
    :pathlib: Path resolution and validation
    :abc: Abstract base class support
    :csv: CSV format writing core
"""
from pathlib import Path
from abc import ABC, abstractmethod
import csv


class Writer(ABC):
    """Abstract base class for different file format writers."""

    @abstractmethod
    def write(self, data: list[tuple]):
        """
        Write a single data row to the output file.

        :param data: Data row to be written.
        :raise OSError: if file system operations fail.
        """
    @abstractmethod
    def clear(self):
        """
        Truncate file to 0 bytes.
        """
    @staticmethod
    def create(output_path: str):
        """
        Factory method to create format-specific Writer instances.
        
        :param output_path: Absolute path to output file
            
        :return Writer: Concrete Writer instance for specified format
            
        :raise ValueError: If path is not absolute or format is unsupported
        :raise FileNotFoundError: If parent directories don't exist
        """
        path = Path(output_path)
        if not path.is_absolute():
            raise ValueError(f"Path must be absolute. Got: {output_path}")
        resolved_path = path.resolve()

        if not resolved_path.parent.exists():
            raise FileNotFoundError(f"Parent directory {resolved_path.parent} does not exist")

        if resolved_path.suffix.lower() == '.csv':
            return CsvWriter(resolved_path)
        raise ValueError(f"Unsupported format: {resolved_path.suffix}")


class CsvWriter(Writer):
    """CSV format writer implementation."""

    def __init__(self, output_path: Path):
        """
        Initialize CSV writer.

        :param output_path: Absolute path to CSV file. 
        """
        self.output_path = output_path
        self.first_write = True

    def write(self, data: list[tuple]):
        """
        Write tuple as CSV row. Automatically handles:
        - File opening on first write
        - String conversion of elements
        - Proper CSV escaping
        """
        try:
            mode = "w" if self.first_write else "a"
            self.first_write = False
            with open(self.output_path, mode, newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows(data)
        except OSError as e:
            raise OSError(f"File system error accessing {self.output_path}: {e}") from e

    def clear(self):
        """
        Truncate file to 0 bytes. 
        """
        try:
            with open(self.output_path, "w", newline="", encoding="utf-8"):
                pass
        except OSError as e:
            raise OSError(f"Failed to clear file {self.output_path}: {e}") from e
