"""
Accuracy Calculator Module tests.
"""

import os
import pytest
from numpy import isclose
from src.accuracy_checker.accuracy_checker import AccuracyCalculator
# import matplotlib.pyplot as plt


@pytest.mark.parametrize(
    "detection_file, expected_tp, expected_fp, expected_fn,", [
        ("detections_perfect.csv", 9, 0, 0),
        ("detections_empty.csv", 0, 0, 9),
        ("detections_fp.csv", 9, 2, 0),
        ("detections_low_iou.csv", 0, 9, 9),
        ("detections_multiple.csv", 5, 3, 4),
        ("detections_wrong_class.csv", 2, 4, 7),
    ])
def test_accuracy_metrics(detection_file, expected_tp, expected_fp, expected_fn):
    """Test TP, FP, FN"""
    gt_file = os.path.abspath("test_data/ground_truth.csv")
    det_file = os.path.abspath(f"test_data/{detection_file}")

    acc_calc = AccuracyCalculator(iou_threshold=0.5)
    acc_calc.load_groundtruths(gt_file)
    acc_calc.load_detections(det_file)

    assert acc_calc.calc_tp() == expected_tp
    assert acc_calc.calc_fp() == expected_fp
    assert acc_calc.calc_fn() == expected_fn


@pytest.mark.parametrize("detection_file", [
    "detections_perfect.csv",
    "detections_fp.csv",
    "detections_low_iou.csv",
    "detections_multiple.csv",
    "detections_wrong_class.csv",
])
def test_precision_recall_curve(detection_file):
    """Generate Precision-Recall graph"""
    gt_file = os.path.abspath("test_data/ground_truth.csv")
    det_file = os.path.abspath(f"test_data/{detection_file}")

    acc_calc = AccuracyCalculator(iou_threshold=0.5)
    acc_calc.load_groundtruths(gt_file)
    acc_calc.load_detections(det_file)

    precisions, recalls = acc_calc.calc_precision_recall("CAR")

    assert len(precisions) == len(recalls)
    assert all(0.0 <= p <= 1.0 for p in precisions)
    assert all(0.0 <= r <= 1.0 for r in recalls)

    # plt.figure()
    # plt.plot(recalls, precisions, marker='o', linestyle='-', color='b')
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.title(f"Precision-Recall Curve ({detection_file})")
    # plt.grid()
    # save_file = os.path.abspath(f"test_results/{detection_file}.png")
    # plt.savefig(save_file)
    # plt.close()


@pytest.mark.parametrize("detection_file,expected_map", [
    ("detections_perfect.csv", 1.0),
    ("detections_fp.csv", 0.875),
    ("detections_low_iou.csv", 0.0),
    ("detections_multiple.csv", 0.31),
    ("detections_wrong_class.csv", 0.21),
])
def test_map(detection_file, expected_map):
    """Test mAP (Mean Average Precision)"""
    gt_file = os.path.abspath("test_data/ground_truth.csv")
    det_file = os.path.abspath(f"test_data/{detection_file}")

    acc_calc = AccuracyCalculator(iou_threshold=0.5)
    acc_calc.load_groundtruths(gt_file)
    acc_calc.load_detections(det_file)

    assert isclose(acc_calc.calc_map(), expected_map, atol=0.01)


@pytest.mark.parametrize("detection_file,expected_tpr,expected_fdr", [
    ("detections_perfect.csv", 1.0, 0.0),
    ("detections_fp.csv", 1.0, 2 / 11),
    ("detections_low_iou.csv", 0.0, 1.0),
    ("detections_multiple.csv", 5 / 9, 3 / 8),
    ("detections_wrong_class.csv", 2 / 9, 4 / 6),
])
def test_tpr_fdr(detection_file, expected_tpr, expected_fdr):
    """Проверка расчёта TPR и FDR"""
    gt_file = os.path.abspath("test_data/ground_truth.csv")
    det_file = os.path.abspath(f"test_data/{detection_file}")

    acc_calc = AccuracyCalculator(iou_threshold=0.5)
    acc_calc.load_groundtruths(gt_file)
    acc_calc.load_detections(det_file)

    tpr = acc_calc.calc_tpr()
    fdr = acc_calc.calc_fdr()

    assert isclose(tpr, expected_tpr, atol=0.01)
    assert isclose(fdr, expected_fdr, atol=0.01)
