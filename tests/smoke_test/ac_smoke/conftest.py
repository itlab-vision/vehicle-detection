import pytest


@pytest.fixture
def valid_gt_csv(tmp_path):
    csv_path = tmp_path / "valid_gt.csv"
    csv_content = "1,CAR,10,10,20,20\n2,BUS,30,30,40,40"
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def valid_detection_csv(tmp_path):
    csv_path = tmp_path / "valid_det.csv"
    csv_content = "1,CAR,10,10,20,20,0.9\n2,BUS,30,30,40,40,0.8"
    csv_path.write_text(csv_content)
    return str(csv_path)
