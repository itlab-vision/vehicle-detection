"""
Fixtures for Data Reader Modules.
"""

import pytest


@pytest.fixture
def valid_gt_csv(tmp_path):
    """
    Creates a temporary valid ground truth CSV file with correctly formatted entries.

    Returns:
        str: Path to the temporary ground truth CSV file.

    Content:
        1,CAR,10,10,20,20
        2,BUS,30,30,40,40
    """
    csv_path = tmp_path / "valid_gt.csv"
    csv_content = "1,CAR,10,10,20,20\n2,BUS,30,30,40,40"
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def invalid_gt_csv(tmp_path):
    """
    Creates a temporary invalid ground truth CSV file with an incorrect format.
    This file contains an extra column that should not be present.

    Returns:
        str: Path to the temporary invalid ground truth CSV file.

    Content:
        1,CAR,10,10,20,20,extra
    """
    csv_path = tmp_path / "invalid_gt.csv"
    csv_content = "1,CAR,10,10,20,20,extra\n"
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def valid_detection_csv(tmp_path):
    """
    Creates a temporary valid detection CSV file with correctly formatted entries.

    Returns:
        str: Path to the temporary detection CSV file.

    Content:
        1,CAR,10,10,20,20,0.9
        2,BUS,30,30,40,40,0.8
    """
    csv_path = tmp_path / "valid_det.csv"
    csv_content = "1,CAR,10,10,20,20,0.9\n2,BUS,30,30,40,40,0.8"
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def invalid_detection_csv(tmp_path):
    """
    Creates a temporary invalid detection CSV file missing the confidence score column.

    Returns:
        str: Path to the temporary invalid detection CSV file.

    Content:
        1,CAR,10,10,20,20  # Missing Confidence column
    """
    csv_path = tmp_path / "invalid_det.csv"
    csv_content = "1,CAR,10,10,20,20\n"  # Missing confidence
    csv_path.write_text(csv_content)
    return str(csv_path)
