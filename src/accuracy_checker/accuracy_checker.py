"""
Accuracy Checker Module

Provides a class for calculating key object detection metrics including:
    - True Positive Rate (TPR)
    - False Detection Rate (FDR)
    - Average Precision (AP)
    - Mean Average Precision (mAP) across multiple classes.

Main Class:
    AccuracyCalculator: Handles metric computation using Intersection over Union (IoU)
    matching between ground truth annotations and detection predictions.
    Supports loading data from CSV files via integrated data readers.

Key Functionality:
    - Loading ground truth annotations and detection results from CSV files
    - Calculation of TP/FP/FN statistics
    - Precision-Recall curve generation
    - Class-specific AP calculation and multi-class mAP aggregation
    - TPR/FDR computation for overall detection performance

Dependencies:
    - data_reader module (GroundtruthReader, DetectionReader) for CSV parsing
    - Built-in Python math operations for IoU calculations
"""

from src.utils.data_reader import CsvGTReader, DetectionReader


class AccuracyCalculator:
    """
    A class that ensures the calculation of the main metrics of the quality detection of objects:
    TPR (True Positive Rate), FDR (False Detection Rate),
    Average Precision (AP) and Mean Average Precision (MAP) according to several classes.
    """

    def __init__(self, iou_threshold=0.5):
        """
        Class initialization for calculating average accuracy (AP).

        :param iou_threshold: The threshold of the International Over Union (IOU) is used to
                            determine whether the found object is consistent with the true markup.
        """
        self.iou_threshold = iou_threshold
        self.groundtruths = {}  # dict: Groundtruths (grouped by classes)
        self.detections = {}    # dict: Detector predictions (grouped by classes)

    def load_groundtruths(self, file_path):
        """
        Downloading groundtruths from the file (.csv).

        :param file_path (str): The path to the file with groundtruths.
        """
        self.groundtruths = self.__format_read_data(CsvGTReader(file_path).read())

    def load_detections(self, file_path:str):
        """
        Loading detections from the file (.csv).

        :param file_path (str): The path to the file with detections.
        """
        self.detections = self.__format_read_data(DetectionReader(file_path).read())

    def calc_tp(self):
        """
        Calculates the total number of True Positive (TP) detections.

        :return: Total number of True Positive detections.
        """
        all_classes = self.groundtruths.keys()
        tp = 0
        for class_name in all_classes:
            detections = self.detections[class_name] if class_name in self.detections else {}
            groundtruths = self.groundtruths[class_name]
            if len(detections) == 0:
                for frame_id, gts in groundtruths.items():
                    tp_det, _, _ = self.__match_dets_to_gts([], gts)
                    tp += tp_det
            else:
                # 1. Sorting predictions by confidence
                all_detections = self.__sort_dets_by_confidence(detections)

                # 2. Search for correspondences between detections and groundtruths
                for frame_id, dets in all_detections.items():
                    gts = groundtruths.get(frame_id, [])    # List of all rectangles for the frame
                    tp_det, _, _ = self.__match_dets_to_gts(dets, gts)
                    tp += tp_det

        return tp

    def calc_fn(self):
        """
        Calculates the total number of False Negative (FN) detections.

        :return: Total number of False Negative detections.
        """
        all_classes = self.groundtruths.keys()
        fn = 0
        for class_name in all_classes:
            detections = self.detections[class_name] if class_name in self.detections else {}
            groundtruths = self.groundtruths[class_name]
            if len(detections) == 0:
                for frame_id, gts in groundtruths.items():
                    _, _, fn_det = self.__match_dets_to_gts([], gts)
                    fn += fn_det
            else:
                # 1. Sorting predictions by confidence
                all_detections = self.__sort_dets_by_confidence(detections)

                # 2. Search for correspondences between detections and groundtruths
                for frame_id, dets in all_detections.items():
                    gts = groundtruths.get(frame_id, [])    # List of all rectangles for the frame
                    _, _, fn_det = self.__match_dets_to_gts(dets, gts)
                    fn += fn_det

        return fn

    def calc_fp(self):
        """
        Calculates the total number of False Positive (FN) detections.

        :return: Total number of False Positive detections.
        """
        all_classes = self.groundtruths.keys()
        fp = 0
        for class_name in all_classes:
            detections = self.detections[class_name] if class_name in self.detections else {}
            groundtruths = self.groundtruths[class_name]
            if len(detections) == 0:
                for frame_id, gts in groundtruths.items():
                    _, fp_det, _ = self.__match_dets_to_gts([], gts)
                    fp += fp_det
            else:
                # 1. Sorting predictions by confidence
                all_detections = self.__sort_dets_by_confidence(detections)

                # 2. Search for correspondences between detections and groundtruths
                for frame_id, dets in all_detections.items():
                    gts = groundtruths.get(frame_id, [])    # List of all rectangles for the frame
                    _, fp_det, _ = self.__match_dets_to_gts(dets, gts)
                    fp += fp_det

        all_classes_dets = self.detections.keys()
        for class_name in all_classes_dets:
            if class_name not in all_classes:
                detections = self.detections[class_name]
                for frame_id, dets in detections.items():
                    fp += len(dets)

        return fp

    def calc_tpr(self):
        """
        Calculates True Positive Rate (TPR).

        :return: True Positive Rate.
        """
        tp = self.calc_tp()
        fn = self.calc_fn()

        return tp / (tp + fn) if (tp + fn) else 0

    def calc_fdr(self):
        """
        Calculates False Detection Rate (FDR).

        :return: False Detection Rate.
        """
        tp = self.calc_tp()
        fp = self.calc_fp()

        return fp / (tp + fp) if (tp + fp) else 0

    def calc_precision_recall(self, class_name:str):
        """
        Calculates Precisions and Recalls.

        :param class_name (str): Class name
        :return: Precisions, Recalls
        """
        if class_name not in self.detections or class_name not in self.groundtruths:
            return [], []

        groundtruths = self.groundtruths[class_name]

        tp_total, fp_total = 0, 0
        fn_total = sum(len(groundtruths.get(frame, [])) for frame in groundtruths)
        precisions_total, recalls_total = [], []

        # Process sorted detections frame-by-frame
        for frame_id, dets in self.__sort_dets_by_confidence(self.detections[class_name]).items():
            gts = groundtruths.get(frame_id, [])  # List of all rectangles for the frame
            tp_det, fp_det, _ = self.__match_dets_to_gts(dets, gts)

            tp_total += tp_det
            fp_total += fp_det
            fn_total -= tp_det

            has_gts = len(gts) > 0
            has_dets = len(dets) > 0

            if has_gts and has_dets:
                precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0
                recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0
            elif (not has_gts) and has_dets:    # tp == 0 nad fp > 0 and fn == 0
                precision = 0
                recall = 1
            elif has_gts and (not has_dets):    # tp == 0 nad fp == 0 and fn > 0
                precision = 1
                recall = 0
            else:                               # tp == 0 nad fp == 0 and fn == 0
                precision = 1
                recall = 1

            precisions_total.append(precision)
            recalls_total.append(recall)

        return precisions_total, recalls_total

    def calc_ap(self, class_name:str):
        """
        Calculates Average Precision (AP).

        :param class_name (str): Class name
        :return: The value of Average Precision (AP).
        """

        precisions, recalls = self.calc_precision_recall(class_name)
        recalls = [0.0] + recalls
        precisions = [2.0] + precisions

        cur_max_prec = precisions[-1]
        rec_end = recalls[-1]
        ap = 0.0
        for i in range(1, len(precisions) + 1):
            rec_start = recalls[-i]

            if precisions[-i] > cur_max_prec:
                ap += (rec_end - rec_start) * cur_max_prec
                cur_max_prec = precisions[-i]
                rec_end = rec_start

        return ap

    def calc_map(self):
        """
        Calculates Mean Average Precision (mAP) for all classes.

        :return: The value of Mean Average Precision (mAP) for all classes.
        """
        all_classes = self.groundtruths.keys()
        total_ap = 0
        for class_name in all_classes:
            total_ap += self.calc_ap(class_name)

        return total_ap / len(all_classes) if all_classes else 0

    # ======= Private methods=======
    @staticmethod
    def __calc_iou(bbox1, bbox2):
        """
        Calculates Intersection over Union (IoU) for two rectangles.

        :param bbox1: The first rectangle [x1, y1, x2, y2].
        :param bbox2: The second rectangle [x1, y1, x2, y2].
        :return: The value of IoU (from 0 to 1).
        """
        xi1, yi1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
        xi2, yi2 = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])

        inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
        bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def __match_dets_to_gts(self, detections, groundtruths):
        """
        Compares detections with groundtruths.

        :param detections: List of detections for the frame.
        :param groundtruths: List of groundtruths for the frame.
        :return: The value of TP (true positives), FP (false positives), FN (false negatives).
        """
        matched = set()
        tp, fp, fn = 0, 0, 0

        for det in detections:
            best_iou = 0
            best_gt_idx = -1
            # Look for the rectangle with the highest iou value
            for idx, gt in enumerate(groundtruths):
                iou = self.__calc_iou(det[:-1], gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= self.iou_threshold and best_gt_idx not in matched:
                tp += 1
                matched.add(best_gt_idx)
            else:
                # Repeat detections are also added here
                fp += 1

        # fn = len(groundtruths) - len(matched) - fp
        fn = len(groundtruths) - len(matched)
        return tp, fp, fn

    @staticmethod
    def __sort_dets_by_confidence(detections):
        """
        Sorts detections by confidence.

        :param detections: Dict {frame_id: [list of detections]}.
        :return: Dict {frame_id: [sorted list of detections]}.
        """
        sorted_detections = {}
        for frame, dets in detections.items():
            # Sort by confidence (last element)
            sorted_detections[frame] = sorted(dets, key=lambda x: -x[-1])

        return sorted_detections

    @staticmethod
    def __format_read_data(data):
        """
        Formats parsed data.

        :param data: Parsed data from CSV file with groundtruths or detections.
        :return: Dict {class_name: {frame_id: [list of bboxes]}}.
        """
        formated_data = {}
        for row in data:
            frame_id, class_name, *args = row
            if class_name not in formated_data:
                formated_data[class_name] = {}
            if frame_id not in formated_data[class_name]:
                formated_data[class_name][frame_id] = []
            formated_data[class_name][frame_id].append(args)

        return formated_data
