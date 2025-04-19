"""
Detection Input/Output Adapters

Provides standardized interfaces for preprocessing inputs and processing raw outputs
from different object detection models.

Functionality:
- Converts input images to model-specific formats
- Transforms model-specific outputs into unified detection format
- Filters detections by confidence threshold
- Applies Non-Maximum Suppression (NMS) for overlapping detections
- Maps numeric class IDs to human-readable names

Classes:
    :Adapter: Abstract base class for detection input/output processing
    :AdapterFasterRCNN: Implementation for Faster R-CNN models
    :AdapterOpenCV: Base adapter for OpenCV-based models
    :AdapterDetectionTask: Adapter for standard OpenCV detection models
    :AdapterYOLO: Adapter for YOLO family models
    :AdapterYOLOTiny: Adapter for YOLO-tiny architectures

Dependencies:
    :cv2: Image processing and NMS operations
    :numpy: Numerical computations
    :abc: Abstract base class support
    :torchvision: PyTorch model transformations
"""
from abc import ABC, abstractmethod
import numpy as np
import cv2 as cv
import torchvision


class Adapter(ABC):
    """
    Abstract adapter class that transforms the detector's input and output into the required format.
    """
    def __init__(self, conf, nms, class_names, interest_classes = None):
        """
        Initializes the adapter with confidence and NMS thresholds.

        :param conf: Confidence threshold for detections.
        :param nms: Non-Maximum Suppression (NMS) threshold.
        :param class_names: List of class names.
        :param interest_classes: List of interest class names.
        """
        if interest_classes is None:
            interest_classes = ['car', 'bus', 'truck']
        self.conf = conf
        self.nms = nms
        self.class_names = class_names
        self.interest_classes = interest_classes

    @abstractmethod
    def pre_processing(self, image: np.ndarray, **kwargs):
        """
        Prepares input image for model inference.

        :param image: Input image in BGR format
        :param kwargs: Additional preprocessing parameters
        :return: Processed input in model-specific format
        """

    @abstractmethod
    def post_processing(self, output: list, image_width: int, image_height: int):
        """
        Transforms output into a readable format.

        :param output: Model output tensor (detections).
        :param image_width: Original image width.
        :param image_height: Original image height.
        :return: List of detections [class, x1, y1, x2, y2, confidence].
        """

    def _nms(self, boxes, confidences, classes_id):
        """
        Applies Non-Maximum Suppression to detection results.

        :param boxes: List of bounding box coordinates
        :param confidences: List of detection confidences
        :param classes_id: List of class identifiers
        :return: Filtered detections after NMS
        """
        indexes = cv.dnn.NMSBoxes(boxes, confidences, self.conf, self.nms)
        bboxes = []
        for i in indexes:
            bboxes.append((classes_id[i], int(boxes[i][0]), int(boxes[i][1]),
                           int(boxes[i][2]), int(boxes[i][3]), confidences[i]))

        return bboxes


class AdapterFasterRCNN(Adapter):
    """
    Adapter implementation for Faster R-CNN models.

    Handles PyTorch-specific preprocessing and output formatting.
    """
    def pre_processing(self, image: np.ndarray, **kwargs):
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_tensor = torchvision.transforms.functional.to_tensor(image_rgb).unsqueeze(0)
        return image_tensor

    def post_processing(self, output: list, image_width: int, image_height: int):
        boxes = output[0]['boxes'].cpu().numpy()
        confidences = output[0]['scores'].cpu().numpy()
        class_labels = output[0]['labels'].cpu().numpy()

        bboxes = []

        for _, (box, confidence, label) in enumerate(zip(boxes, confidences, class_labels)):
            if confidence > self.conf:
                # Get class name based on label index
                class_name = self.class_names[label.item()]
                if class_name in self.interest_classes:
                    bboxes.append([class_name, *[int(val) for val in box], confidence])

        return self.__apply_nms(bboxes)

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
        indexes = cv.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), self.conf, self.nms)
        if len(indexes) == 0:
            return []

        return [detections[i] for i in indexes]


class AdapterOpenCV(Adapter, ABC):
    """
    Base adapter class for OpenCV DNN models.

    Implements blob-based preprocessing using cv2.dnn.blobFromImage.
    """
    def pre_processing(self, image: np.ndarray, **kwargs):
        """
        Creates input blob for OpenCV models.

        :param image: Input image in BGR format
        :param kwargs: blobFromImage parameters:
            - scalefactor: Scale multiplier
            - size: Spatial dimensions for output blob
            - mean: Mean subtraction values
            - swapRB: Flag for BGR to RGB conversion
        :return: Formatted input blob
        """
        return cv.dnn.blobFromImage(
            image=image,
            scalefactor=kwargs['scalefactor'],
            size=kwargs['size'],
            mean=kwargs['mean'],
            swapRB=kwargs['swapRB']
        )


class AdapterDetectionTask(AdapterOpenCV):
    """
    Adapter for standard OpenCV detection models.
    """
    def post_processing(self, output, image_width, image_height):
        classes_id = []
        confidences = []
        boxes = []
        for i in range(output.shape[2]):
            box = output[0, 0, i]
            confidence = box[2]
            if confidence > self.conf:

                left = min(int(box[3] * image_width), image_width)
                top = min(int(box[4] * image_height), image_height)
                right = min(int(box[5] * image_width), image_width)
                bottom = min(int(box[6] * image_height), image_height)

                class_name = self.class_names[int(box[1])]

                if class_name in self.interest_classes:
                    boxes.append((left, top, right, bottom))
                    classes_id.append(class_name)
                    confidences.append(confidence)

        return self._nms(boxes, confidences, classes_id)


class AdapterYOLO(AdapterOpenCV):
    """
    Adapter for YOLO models.
    """
    def post_processing(self, output, image_width, image_height):
        classes_id = []
        boxes = []
        confidences = []
        for detection in output:

            scores = detection[5:]
            confidence = scores[np.argmax(scores)]
            class_name = self.class_names[np.argmax(scores)]
            if confidence > self.conf:

                cx1 = int(detection[0] * image_width)
                cy1 = int(detection[1] * image_height)
                w = int(detection[2] * image_width)
                h = int(detection[3] * image_height)

                if class_name in self.interest_classes:
                    boxes.append((cx1 - w // 2, cy1 - h // 2,
                                  cx1 + w // 2, cy1 + h // 2))
                    classes_id.append(class_name)
                    confidences.append(confidence)

        return self._nms(boxes, confidences, classes_id)


class AdapterYOLOTiny(AdapterOpenCV):
    """
    Adapter for YOLO-tiny models with grid-based output decoding.
    """
    def __demo_postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            grid = np.stack((np.meshgrid(np.arange(wsize), np.arange(hsize))), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        return outputs

    def post_processing(self, output, image_width, image_height):
        predictions = self.__demo_postprocess(output[0], (416, 416))
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        b_xyxy = np.ones_like(boxes)

        b_xyxy[:, 0] = boxes[:, 0] / (416 / image_width) - boxes[:, 2]/2. / (416 / image_width)
        b_xyxy[:, 1] = boxes[:, 1] / (416 / image_height) - boxes[:, 3]/2. / (416 / image_height)
        b_xyxy[:, 2] = boxes[:, 0] / (416 / image_width) + boxes[:, 2]/2. / (416 / image_width)
        b_xyxy[:, 3] = boxes[:, 1] / (416 / image_height) + boxes[:, 3]/2. / (416 / image_height)

        all_classes_id = scores.argmax(1)
        all_confidences = scores[np.arange(len(all_classes_id)), all_classes_id]

        classes_id = []
        boxes = []
        confidences = []
        for i, class_id in zip(range(len(all_classes_id)), all_classes_id):
            if self.class_names[class_id] in self.interest_classes:
                classes_id.append(self.class_names[class_id])
                boxes.append(b_xyxy[i])
                confidences.append(all_confidences[i])

        return self._nms(boxes, confidences, classes_id)
