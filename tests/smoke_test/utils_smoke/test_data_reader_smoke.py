from src.utils.data_reader import CsvGTReader, DetectionReader


def test_csv_gt_reader_valid_file(valid_gt_csv):
    reader = CsvGTReader(valid_gt_csv)
    data = reader.read()
    assert len(data) == 2
    assert data[0] == (1, 'CAR', 10, 10, 20, 20)
    assert data[1] == (2, 'BUS', 30, 30, 40, 40)

def test_csv_gt_reader_invalid_line(invalid_gt_csv):
    reader = CsvGTReader(invalid_gt_csv)
    data = reader.read()
    assert len(data) == 0  # Пропущена некорректная строка

def test_detection_reader_valid_file(valid_detection_csv):
    reader = DetectionReader(valid_detection_csv)
    data = reader.read()
    assert len(data) == 2
    assert data[0] == (1, 'CAR', 10, 10, 20, 20, 0.9)
    assert data[1] == (2, 'BUS', 30, 30, 40, 40, 0.8)

def test_detection_reader_invalid_line(invalid_detection_csv):
    reader = DetectionReader(invalid_detection_csv)
    data = reader.read()
    assert len(data) == 0  # Пропущена строка с ошибкой
