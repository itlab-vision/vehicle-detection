"""
Detection Output Adapters

Provides a standardized interface for processing raw outputs from different object detection models.

Functionality:
- Converts model-specific output formats into a unified list of detections.
- Filters detections by confidence threshold.
- Supports Non-Maximum Suppression (NMS) for reducing redundant detections.
- Maps numerical class labels to human-readable class names.

Classes:
    :Adapter: Abstract base class defining the interface for detection output processing.
    :AdapterFasterRCNN: Concrete implementation for processing Faster R-CNN outputs.

Dependencies:
    :cv2: for image handling and Non-Maximum Suppression (NMS)
    :numpy: for numerical operations
    :abc: for abstract base class support
"""

from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np


class Adapter(ABC):
    """
    Abstract adapter class that transforms the detector's output into the required format.
    """

    def __init__(self, conf_threshold: float, nms_threshold: float, class_names: list):
        """
        Initializes the adapter with confidence and NMS thresholds.

        :param conf_threshold: Confidence threshold for detections.
        :param nms_threshold: Non-Maximum Suppression (NMS) threshold.
        :param class_names: List of class names.
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.class_names = class_names

    @abstractmethod
    def post_processing(self, output: list, image_width: int, image_height: int):
        """
        Processes the model output and converts it to [class, x1, y1, x2, y2, confidence] format.

        :param output: Raw model output (detections).
        :param image_width: Width of the original image.
        :param image_height: Height of the original image.
        :return: List of detections in [class, x1, y1, x2, y2, confidence] format.
        """


class AdapterFasterRCNN(Adapter):
    """
    Adapter for processing Faster R-CNN model output.
    """

    def __init__(self, conf_threshold: float, nms_threshold: float, class_names: list):
        """
        Initializes the adapter with confidence and NMS thresholds.

        :param conf_threshold: Confidence threshold for detections.
        :param nms_threshold: Non-Maximum Suppression (NMS) threshold.
        :param class_names: List of class names to detect (e.g., ['car', 'bus']).
        """
        super().__init__(conf_threshold, nms_threshold, class_names)

        # Predefined class labels from COCO dataset
        self.coco_classes = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter',
            'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'handbag', 'tie', 'suitcase',
            'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat','baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'N/A', 'dining table', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def post_processing(self, output: list, image_width: int, image_height: int):
        """
        Transforms Faster R-CNN output into a readable format.

        :param output: Model output tensor (detections).
        :param image_width: Original image width.
        :param image_height: Original image height.
        :return: List of detections [class, x1, y1, x2, y2, confidence].
        """
        boxes = output[0]['boxes'].cpu().numpy()
        confidences = output[0]['scores'].cpu().numpy()
        class_labels = output[0]['labels'].cpu().numpy()

        results = []

        for _, (box, confidence, label) in enumerate(zip(boxes, confidences, class_labels)):
            if confidence > self.conf_threshold:
                # Get class name based on label index
                class_name = self.coco_classes[label.item()]
                if class_name in self.class_names:
                    results.append([class_name, *[int(val) for val in box], confidence])

        results = self.__apply_nms(results)
        return results

    def __apply_nms(self, detections: list):
        """
        Apply Non-Maximum Suppression (NMS) to remove redundant detections.

        :param detections: List of detections [class, x1, y1, x2, y2, confidence].
        :return: List of detections after NMS.
        """
        if len(detections) == 0:
            return []

        boxes = np.array([det[1:5] for det in detections])
        confidences = np.array([det[5] for det in detections])
        indexes = cv.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(),
                                  self.conf_threshold, self.nms_threshold)
        if len(indexes) == 0:
            return []

        return [detections[i] for i in indexes]
