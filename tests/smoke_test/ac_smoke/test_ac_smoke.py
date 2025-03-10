"""
Accuracy Calculator Module tests.
"""

import os
import pytest
from numpy import isclose
from src.accuracy_checker.accuracy_checker import AccuracyCalculator


@pytest.mark.parametrize(
        "detection_file, expected_tp, expected_fp, expected_fn,", [
        ("detections_perfect.csv", 9, 0, 0),
        ("detections_empty.csv", 0, 0, 9),
        ("detections_fp.csv", 9, 2, 0),
        ("detections_low_iou.csv", 0, 9, 9),
        ("detections_multiple.csv", 5, 3, 4),
        ("detections_wrong_class.csv", 2, 4, 7),
    ])
def test_small_data_accuracy_metrics(detection_file, expected_tp, expected_fp, expected_fn):
    """Test TP, FP, FN on small data"""
    gt_file = os.path.abspath("test_data/small_data/ground_truth.csv")
    det_file = os.path.abspath(f"test_data/small_data/{detection_file}")

    acc_calc = AccuracyCalculator(iou_threshold=0.5)
    acc_calc.load_groundtruths(gt_file)
    acc_calc.load_detections(det_file)

    assert acc_calc.calc_total_tp() == expected_tp
    assert acc_calc.calc_total_fp() == expected_fp
    assert acc_calc.calc_total_fn() == expected_fn


@pytest.mark.parametrize(
        "detection_file, expected_tpr, expected_fdr", [
        ("detections_perfect.csv", 1.0, 0.0),
        ("detections_fp.csv", 1.0, 2 / 11),
        ("detections_low_iou.csv", 0.0, 1.0),
        ("detections_multiple.csv", 5 / 9, 3 / 8),
        ("detections_wrong_class.csv", 2 / 9, 4 / 6),
    ])
def test_small_data_tpr_fdr(detection_file, expected_tpr, expected_fdr):
    """Test TPR and FDR on small data"""
    gt_file = os.path.abspath("test_data/small_data/ground_truth.csv")
    det_file = os.path.abspath(f"test_data/small_data/{detection_file}")

    acc_calc = AccuracyCalculator(iou_threshold=0.5)
    acc_calc.load_groundtruths(gt_file)
    acc_calc.load_detections(det_file)

    tpr = acc_calc.calc_tpr()
    fdr = acc_calc.calc_fdr()

    assert isclose(tpr, expected_tpr, atol=0.01)
    assert isclose(fdr, expected_fdr, atol=0.01)


@pytest.mark.parametrize(
        "detection_file", [
        "detections_perfect.csv",
        "detections_fp.csv",
        "detections_low_iou.csv",
        "detections_multiple.csv",
        "detections_wrong_class.csv",
    ])
def test_small_data_precision_recall_curve(detection_file):
    """Generate Precision-Recall graph on small data"""
    gt_file = os.path.abspath("test_data/small_data/ground_truth.csv")
    det_file = os.path.abspath(f"test_data/small_data/{detection_file}")

    acc_calc = AccuracyCalculator(iou_threshold=0.5)
    acc_calc.load_groundtruths(gt_file)
    acc_calc.load_detections(det_file)

    precisions, recalls = acc_calc.calc_precision_recall("CAR")

    assert len(precisions) == len(recalls)
    assert all(0.0 <= p <= 1.0 for p in precisions)
    assert all(0.0 <= r <= 1.0 for r in recalls)


@pytest.mark.parametrize(
        "detection_file, expected_ap_car, expected_ap_bus", [
        ("detections_perfect.csv", 1.0, 1.0),
        ("detections_fp.csv", 1.0, 1.0),
        ("detections_low_iou.csv", 0.0, 0.0),
        ("detections_multiple.csv", 0.67, 0.33),
        ("detections_wrong_class.csv", 0.17, 0.33),
    ])
def test_small_data_ap(detection_file, expected_ap_car, expected_ap_bus):
    """Test AP (Average Precision) on small data"""
    gt_file = os.path.abspath("test_data/small_data/ground_truth.csv")
    det_file = os.path.abspath(f"test_data/small_data/{detection_file}")

    acc_calc = AccuracyCalculator(iou_threshold=0.5)
    acc_calc.load_groundtruths(gt_file)
    acc_calc.load_detections(det_file)

    assert isclose(acc_calc.calc_ap("CAR"), expected_ap_car, atol=0.01)
    assert isclose(acc_calc.calc_ap("BUS"), expected_ap_bus, atol=0.01)


@pytest.mark.parametrize(
        "detection_file, expected_map", [
        ("detections_perfect.csv", 1.0),
        ("detections_fp.csv", 1.0),
        ("detections_low_iou.csv", 0.0),
        ("detections_multiple.csv", 0.5),
        ("detections_wrong_class.csv", 0.25),
    ])
def test_small_data_map(detection_file, expected_map):
    """Test mAP (Mean Average Precision) on small data"""
    gt_file = os.path.abspath("test_data/small_data/ground_truth.csv")
    det_file = os.path.abspath(f"test_data/small_data/{detection_file}")

    acc_calc = AccuracyCalculator(iou_threshold=0.5)
    acc_calc.load_groundtruths(gt_file)
    acc_calc.load_detections(det_file)

    assert isclose(acc_calc.calc_map(), expected_map, atol=0.01)


@pytest.mark.parametrize(
        "detection_file, expected_tp, expected_fp, expected_fn,", [
        ("detections_regular.csv", 1145, 45, 51),
        ("detections_fp.csv", 1100, 244, 95),
        ("detections_low_iou.csv", 611, 572, 576),
        ("detections_multiple.csv", 1153, 426, 45),
        ("detections_wrong_class.csv", 769, 423, 215)
    ])
def test_big_data_accuracy_metrics(detection_file, expected_tp, expected_fp, expected_fn):
    """Test TP, FP, FN on big data"""
    gt_file = os.path.abspath("test_data/big_data/ground_truth.csv")
    det_file = os.path.abspath(f"test_data/big_data/{detection_file}")

    acc_calc = AccuracyCalculator(iou_threshold=0.5)
    acc_calc.load_groundtruths(gt_file)
    acc_calc.load_detections(det_file)

    assert acc_calc.calc_total_tp() == expected_tp
    assert acc_calc.calc_total_fp() == expected_fp
    assert acc_calc.calc_total_fn() == expected_fn


@pytest.mark.parametrize(
        "detection_file, expected_tpr, expected_fdr", [
        ("detections_regular.csv", 0.95, 0.04),
        ("detections_fp.csv", 0.92, 0.18),
        ("detections_low_iou.csv", 0.51, 0.48),
        ("detections_multiple.csv", 0.96, 0.27),
        ("detections_wrong_class.csv", 0.78, 0.35),
    ])
def test_big_data_tpr_fdr(detection_file, expected_tpr, expected_fdr):
    """Test TPR and FDR on big data"""
    gt_file = os.path.abspath("test_data/big_data/ground_truth.csv")
    det_file = os.path.abspath(f"test_data/big_data/{detection_file}")

    acc_calc = AccuracyCalculator(iou_threshold=0.5)
    acc_calc.load_groundtruths(gt_file)
    acc_calc.load_detections(det_file)

    tpr = acc_calc.calc_tpr()
    fdr = acc_calc.calc_fdr()

    assert isclose(tpr, expected_tpr, atol=0.01)
    assert isclose(fdr, expected_fdr, atol=0.01)


@pytest.mark.parametrize(
        "detection_file", [
        "detections_regular.csv",
        "detections_fp.csv",
        "detections_low_iou.csv",
        "detections_multiple.csv",
        "detections_wrong_class.csv"
    ])
def test_big_data_precision_recall_curve(detection_file):
    """Generate Precision-Recall graph on big data"""
    gt_file = os.path.abspath("test_data/big_data/ground_truth.csv")
    det_file = os.path.abspath(f"test_data/big_data/{detection_file}")

    acc_calc = AccuracyCalculator(iou_threshold=0.5)
    acc_calc.load_groundtruths(gt_file)
    acc_calc.load_detections(det_file)

    precisions, recalls = acc_calc.calc_precision_recall("CAR")

    assert len(precisions) == len(recalls)
    assert all(0.0 <= p <= 1.0 for p in precisions)
    assert all(0.0 <= r <= 1.0 for r in recalls)


@pytest.mark.parametrize(
        "detection_file, expected_ap_car, expected_ap_bus", [
        ("detections_regular.csv", 0.93, 0.98),
        ("detections_fp.csv", 0.84, 0.96),
        ("detections_low_iou.csv", 0.23, 0.76),
        ("detections_multiple.csv", 0.82, 0.87),
        ("detections_wrong_class.csv", 0.61, 0.72),
    ])
def test_big_data_ap(detection_file, expected_ap_car, expected_ap_bus):
    """Test AP (Average Precision) on big data"""
    gt_file = os.path.abspath("test_data/big_data/ground_truth.csv")
    det_file = os.path.abspath(f"test_data/big_data/{detection_file}")

    acc_calc = AccuracyCalculator(iou_threshold=0.5)
    acc_calc.load_groundtruths(gt_file)
    acc_calc.load_detections(det_file)

    assert isclose(acc_calc.calc_ap("CAR"), expected_ap_car, atol=0.01)
    assert isclose(acc_calc.calc_ap("BUS"), expected_ap_bus, atol=0.01)


@pytest.mark.parametrize(
        "detection_file, expected_map", [
        ("detections_regular.csv", 0.95),
        ("detections_fp.csv", 0.9),
        ("detections_low_iou.csv", 0.51),
        ("detections_multiple.csv", 0.85),
        ("detections_wrong_class.csv", 0.67),
    ])
def test_big_data_map(detection_file, expected_map):
    """Test mAP (Mean Average Precision) on big data"""
    gt_file = os.path.abspath("test_data/big_data/ground_truth.csv")
    det_file = os.path.abspath(f"test_data/big_data/{detection_file}")

    acc_calc = AccuracyCalculator(iou_threshold=0.5)
    acc_calc.load_groundtruths(gt_file)
    acc_calc.load_detections(det_file)

    assert isclose(acc_calc.calc_map(), expected_map, atol=0.01)
