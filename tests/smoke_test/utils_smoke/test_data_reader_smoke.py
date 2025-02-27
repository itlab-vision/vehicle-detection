"""
Data Reader Modules tests.
"""

from src.utils.data_reader import CsvGTReader, DetectionReader


def test_csv_gt_reader_valid_file(valid_gt_csv):
    """
    Test that CsvGTReader correctly reads a valid ground truth CSV file.

    Args:
        valid_gt_csv (str): Path to a valid ground truth CSV file.

    Asserts:
        - The reader correctly reads two entries.
        - The first entry matches the expected ground truth values.
        - The second entry matches the expected ground truth values.
    """
    reader = CsvGTReader(valid_gt_csv)
    data = reader.read()
    assert len(data) == 2
    assert data[0] == (1, 'CAR', 10, 10, 20, 20)
    assert data[1] == (2, 'BUS', 30, 30, 40, 40)

def test_csv_gt_reader_invalid_line(invalid_gt_csv):
    """
    Test that CsvGTReader correctly handles an invalid ground truth CSV file.

    Args:
        invalid_gt_csv (str): Path to an invalid ground truth CSV file.

    Asserts:
        - The reader skips the malformed line and returns an empty list.
    """
    reader = CsvGTReader(invalid_gt_csv)
    data = reader.read()
    assert len(data) == 0

def test_detection_reader_valid_file(valid_detection_csv):
    """
    Test that DetectionReader correctly reads a valid detection CSV file.

    Args:
        valid_detection_csv (str): Path to a valid detection CSV file.

    Asserts:
        - The reader correctly reads two entries.
        - The first entry matches the expected detection values.
        - The second entry matches the expected detection values.
    """
    reader = DetectionReader(valid_detection_csv)
    data = reader.read()
    assert len(data) == 2
    assert data[0] == (1, 'CAR', 10, 10, 20, 20, 0.9)
    assert data[1] == (2, 'BUS', 30, 30, 40, 40, 0.8)

def test_detection_reader_invalid_line(invalid_detection_csv):
    """
    Test that DetectionReader correctly handles an invalid detection CSV file.

    Args:
        invalid_detection_csv (str): Path to an invalid detection CSV file.

    Asserts:
        - The reader skips the malformed line and returns an empty list.
    """
    reader = DetectionReader(invalid_detection_csv)
    data = reader.read()
    assert len(data) == 0
