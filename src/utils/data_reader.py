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


class GroundtruthReader:
    """
    A utility class for reading and parsing groundtruth data from a CSV file.

    The CSV file should have the following format:
        frame_id, class_name, x1, y1, x2, y2

    - `frame_id` (int): The frame number.
    - `class_name` (str): The object class.
    - `x1, y1, x2, y2` (int): Bounding box coordinates.
    """

    @staticmethod
    def read(file_path):
        """
        Parsing CSV file with groundtruths.

        :param file_path: The path to the file with groundtruths.
        :return: list[tuples] of parsed data by rows.
        """
        parsed_data = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
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
            print(f"File {file_path} was not found.")
        except Exception as e:
            print(f"Error when reading the file {file_path}: {e}")

        return parsed_data


class DetectionReader:
    """
    A utility class for reading and parsing detections data from a CSV file.

    The CSV file should have the following format:
        frame_id, class_name, x1, y1, x2, y2, confidence

    - `frame_id` (int): The frame number.
    - `class_name` (str): The object class.
    - `x1, y1, x2, y2` (int): Bounding box coordinates.
    - `confidence` (float): A confidence score.
    """

    @staticmethod
    def read(file_path):
        """
        Parsing CSV file with detections.

        :param file_path: The path to the file with detections.
        :return: list[tuples] of parsed data by rows.
        """
        parsed_data = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
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
            print(f"File {file_path} was not found.")
        except Exception as e:
            print(f"Error when reading the file {file_path}: {e}")

        return parsed_data
