"""
Ground Truth and Detection Data Readers Module

Provides abstract and concrete implementations for reading annotation data from:
- CSV files with ground truth
- CSV files with detection results
- Synthetic data generation for testing

Classes:
    GroundtruthReader: Abstract base class for data readers
    CsvGTReader: CSV parser for ground truth annotations
    FakeGTReader: Synthetic data generator for testing
    DetectionReader: CSV parser for detection results with confidence scores

Dependencies:
    - csv module for file parsing
    - random module for synthetic data generation
"""

import csv
from abc import ABC, abstractmethod


class DataReader(ABC):
    """
    Abstract base class for data reading implementations.

    :param filepath: Path to data source file.
    """

    def __init__(self, filepath):
        self.file_path = filepath

    @abstractmethod
    def read(self):
        """Parse file and return structured annotation data"""

class CsvGTReader(DataReader):
    """
    A utility class for reading and parsing groundtruth data from a CSV file.

    The CSV file should have the following format:
        frame_id, class_name, x1, y1, x2, y2

    - `frame_id` (int): The frame number.
    - `class_name` (str): The object class.
    - `x1, y1, x2, y2` (int): Bounding box coordinates.
    """

    def read(self):
        """
        Parsing CSV file with groundtruths.

        :return: list[tuples] of parsed data by rows.
        """
        parsed_data = []
        try:
            with open(self.file_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) != 6:
                        print(f"Incorrect line in the file: {row}")
                        continue

                    frame_id, class_name, x1, y1, x2, y2 = row
                    row_data = (int(frame_id), str(class_name), int(x1), int(y1),
                                int(x2), int(y2))
                    parsed_data.append(row_data)

        except FileNotFoundError:
            print(f"File {self.file_path} was not found.")
        except (ValueError, csv.Error) as e:
            print(f"Data format error in {self.file_path}: {e}")
        except OSError as e:
            print(f"File system error accessing {self.file_path}: {e}")

        return parsed_data


class DetectionReader(DataReader):
    """
    A utility class for reading and parsing detections data from a CSV file.

    The CSV file should have the following format:
        frame_id, class_name, x1, y1, x2, y2, confidence

    - `frame_id` (int): The frame number.
    - `class_name` (str): The object class.
    - `x1, y1, x2, y2` (int): Bounding box coordinates.
    - `confidence` (float): A confidence score.
    """

    def read(self):
        """
        Parsing CSV file with detections.

        :return: list[tuples] of parsed data by rows.
        """
        parsed_data = []
        try:
            with open(self.file_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) != 7:
                        print(f"Incorrect line in the file: {row}")
                        continue

                    frame_id, class_name, x1, y1, x2, y2, confidence = row
                    row_data = (int(frame_id), str(class_name), int(x1), int(y1),
                                int(x2), int(y2), float(confidence))
                    parsed_data.append(row_data)

        except FileNotFoundError:
            print(f"File {self.file_path} was not found.")
        except (ValueError, csv.Error) as e:
            print(f"Data format error in {self.file_path}: {e}")
        except OSError as e:
            print(f"File system error accessing {self.file_path}: {e}")

        return parsed_data
