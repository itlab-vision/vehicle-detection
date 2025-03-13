"""
Data Reader Modules tests.
"""

import pytest
from src.utils.data_reader import CsvGTReader, DetectionReader


def test_csv_gt_reader_valid_file(valid_gt_csv):
    """
    Test that CsvGTReader correctly reads a valid ground truth CSV file.
    """
    reader = CsvGTReader(valid_gt_csv)
    data = reader.read()
    assert len(data) == 2
    assert data[0] == (1, 'CAR', 10, 10, 20, 20)
    assert data[1] == (2, 'BUS', 30, 30, 40, 40)


def test_csv_gt_reader_invalid_line(invalid_gt_csv):
    """
    Test that CsvGTReader correctly raises ValueError on an invalid ground truth CSV file.
    """
    with pytest.raises(ValueError):
        reader = CsvGTReader(invalid_gt_csv)
        reader.read()


def test_detection_reader_valid_file(valid_detection_csv):
    """
    Test that DetectionReader correctly reads a valid detection CSV file.
    """
    reader = DetectionReader(valid_detection_csv)
    data = reader.read()
    assert len(data) == 2
    assert data[0] == (1, 'CAR', 10, 10, 20, 20, 0.9)
    assert data[1] == (2, 'BUS', 30, 30, 40, 40, 0.8)


def test_detection_reader_invalid_line(invalid_detection_csv):
    """
    Test that DetectionReader correctly raises ValueError on an invalid detection CSV file.
    """
    with pytest.raises(ValueError):
        reader = DetectionReader(invalid_detection_csv)
        reader.read()
