"""
Fixtures for Data Reader Modules.
"""

import os
import pytest
from pathlib import Path


@pytest.fixture
def valid_gt_csv(tmp_path):
    """
    Creates a temporary valid ground truth CSV file with correctly formatted entries.

    Content:
        1,CAR,10,10,20,20
        2,BUS,30,30,40,40

    :return: Path to the temporary ground truth CSV file.
    """
    str_path = os.path.join(tmp_path, "valid_gt.csv")
    csv_path = Path(str_path)

    csv_content = "1,CAR,10,10,20,20\n2,BUS,30,30,40,40"
    csv_path.write_text(csv_content)

    return str_path


@pytest.fixture
def invalid_gt_csv(tmp_path):
    """
    Creates a temporary invalid ground truth CSV file with an incorrect format.
    This file contains an extra column that should not be present.

    Content:
        1,CAR,10,10,20,20,extra

    :return: Path to the temporary invalid ground truth CSV file.
    """
    str_path = os.path.join(tmp_path, "invalid_gt.csv")
    csv_path = Path(str_path)

    csv_content = "1,CAR,10,10,20,20,extra\n"
    csv_path.write_text(csv_content)

    return str_path


@pytest.fixture
def valid_detection_csv(tmp_path):
    """
    Creates a temporary valid detection CSV file with correctly formatted entries.

    Content:
        1,CAR,10,10,20,20,0.9
        2,BUS,30,30,40,40,0.8

    :return: Path to the temporary detection CSV file.
    """
    str_path = os.path.join(tmp_path, "valid_det.csv")
    csv_path = Path(str_path)

    csv_content = "1,CAR,10,10,20,20,0.9\n2,BUS,30,30,40,40,0.8"
    csv_path.write_text(csv_content)

    return str_path


@pytest.fixture
def invalid_detection_csv(tmp_path):
    """
    Creates a temporary invalid detection CSV file missing the confidence score column.

    Content:
        1,CAR,10,10,20,20

    :return: Path to the temporary invalid detection CSV file.
    """
    str_path = os.path.join(tmp_path, "invalid_det.csv")
    csv_path = Path(str_path)

    csv_content = "1,CAR,10,10,20,20\n"
    csv_path.write_text(csv_content)

    return str_path
