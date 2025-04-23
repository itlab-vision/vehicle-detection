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
import random
from abc import ABC, abstractmethod

import cv2
import numpy as np
import cv2 as cv
import torch
import torchvision
from scipy import special
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops


class Adapter(ABC):
    """
    Abstract adapter class that transforms the detector's input and output into the required format.
    """
    def __init__(self, conf, nms, class_names, interest_classes=None):
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
    def pre_processing(self, images: np.ndarray, **kwargs):
        """
        Prepares input image for model inference.

        :param images: Input image in BGR format
        :param kwargs: Additional preprocessing parameters
        :return: Processed input in model-specific format
        """

    @abstractmethod
    def post_processing(self, outputs: list, image_sizes: list, **kwargs):
        """
        Transforms output into a readable format.

        :param image_sizes: list[(width, height)]
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
    def pre_processing(self, images: np.ndarray, **kwargs):
        image_tensors = []
        for image in images:
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            tensor = torchvision.transforms.functional.to_tensor(image_rgb)
            image_tensors.append(tensor)
        return image_tensors

    def post_processing(self, outputs: list, image_sizes: list, **kwargs):
        batch_detections = []
        for output, (_, _) in zip(outputs, image_sizes):
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            detections = []
            for box, score, label in zip(boxes, scores, labels):
                if score >= self.conf:
                    class_name = self.class_names[label - 1]
                    if class_name in self.interest_classes:
                        detections.append(
                            [class_name, int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                             float(score)])

            detections = self.__apply_nms(detections)
            batch_detections.append(detections)
        return batch_detections

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
    def pre_processing(self, images: list[np.ndarray], **kwargs):
        """
        Prepares images for OpenCV DNN models by resizing, normalizing,
        and optionally swapping channels.

        :param images: List of input images in BGR format
        :param kwargs: Preprocessing parameters:
            - scalefactor: Scale multiplier (e.g., 1/255.0 for normalization)
            - size: Spatial dimensions for the output blob (e.g., (640, 640))
            - mean: Mean subtraction values as a tuple (e.g., (0, 0, 0))
            - swapRB: Boolean flag to convert BGR to RGB
        :return: List of processed images (numpy arrays)
        """
        res_images = []
        for image in images:
            tmp = cv.resize(image, kwargs['size'])

            # Normalize the image (subtract mean values)
            tmp = (tmp - np.array(kwargs['mean'])) * kwargs['scalefactor']

            # Swap R and B channels if necessary
            if kwargs['swapRB']:
                tmp = tmp[:, :, ::-1]  # BGR -> RGB

            res_images.append(tmp)

        return res_images


# need testing of batch processing implementation
class AdapterDetectionTask(AdapterOpenCV):
    """
    Adapter for standard OpenCV detection models.
    """

    def post_processing(self, outputs: list, image_sizes: list, **kwargs):
        batch_detections = []
        for output, (img_w, img_h) in zip(outputs, image_sizes):
            batch_detections.append(self._process_single_output(output, img_w, img_h))
        return batch_detections

    def _process_single_output(self, output, image_width, image_height):
        detections = []
        for i in range(output.shape[2]):
            detection = self._extract_detection(output[0, 0, i], image_width, image_height)
            if detection:
                detections.append(detection)
        return self._process_detections(detections)

    def _extract_detection(self, detection, img_w, img_h):
        confidence = detection[2]
        if confidence <= self.conf:
            return None

        class_id = int(detection[1])
        class_name = self.class_names[class_id]
        if class_name not in self.interest_classes:
            return None

        return (*self._calculate_coordinates(detection[3:7], img_w, img_h),
                class_name, float(confidence))

    def _process_detections(self, detections):
        boxes, confidences, classes = self._split_detections(detections)
        return self._nms(boxes, confidences, classes)

    @staticmethod
    def _calculate_coordinates(coords, img_w, img_h):
        return (
            int(coords[0] * img_w),
            int(coords[1] * img_h),
            int(coords[2] * img_w),
            int(coords[3] * img_h)
        )

    @staticmethod
    def _split_detections(detections):
        return zip(*[(d[0:4], d[4], d[5]) for d in detections]) if detections else ([], [], [])


# need testing of batch processing implementation
class AdapterYOLO(AdapterOpenCV):
    """
    Adapter for YOLO models.
    """

    def post_processing(self, outputs: list, image_sizes: list, **kwargs):
        batch_detections = []
        for output, (img_w, img_h) in zip(outputs, image_sizes):
            batch_detections.append(self._process_single_output(output[0], img_w, img_h))
        return batch_detections

    def _process_single_output(self, output, img_w, img_h):
        detections = [self._parse_detection(d, img_w, img_h) for d in output]
        valid_detections = [d for d in detections if d is not None]
        return self._process_valid_detections(valid_detections)

    def _parse_detection(self, detection, img_w, img_h):
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence <= self.conf:
            return None

        class_name = self.class_names[class_id]
        if class_name not in self.interest_classes:
            return None

        return *self._calculate_coordinates(detection[:4], img_w, img_h), class_name, confidence

    @staticmethod
    def _calculate_coordinates(coords, img_w, img_h):
        cx, cy, w, h = coords
        return (
            int((cx - w/2) * img_w),
            int((cy - h/2) * img_h),
            int((cx + w/2) * img_w),
            int((cy + h/2) * img_h)
        )

    def _process_valid_detections(self, detections):
        boxes, confidences, classes = zip(*[(d[0:4], d[4], d[5]) for d in detections]) if (
            detections) else ([], [], [])
        return self._nms(boxes, confidences, classes)


# need testing of batch processing implementation
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

    def post_processing(self, outputs: list, image_sizes: list, **kwargs):
        batch_detections = []
        for output, (img_w, img_h) in zip(outputs, image_sizes):
            batch_detections.append(self._process_single_output(output[0], img_w, img_h))
        return batch_detections

    def _process_single_output(self, output, img_w, img_h):
        predictions = self.__demo_postprocess(output, (416, 416))
        scaled_boxes = self._scale_boxes(predictions[:, :4], img_w, img_h)
        class_ids, confidences = self._get_class_info(predictions[:, 4:])
        return self._filter_and_process(scaled_boxes, confidences, class_ids)

    def _scale_boxes(self, boxes, img_w, img_h):
        return [self._scale_single_box(box, img_w, img_h) for box in boxes]

    @staticmethod
    def _scale_single_box(box, img_w, img_h):
        x_center, y_center, width, height = box
        return (
            int((x_center - width/2) / (416 / img_w)),
            int((y_center - height/2) / (416 / img_h)),
            int((x_center + width/2) / (416 / img_w)),
            int((y_center + height/2) / (416 / img_h))
        )

    def _get_class_info(self, scores):
        class_ids = scores.argmax(axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        return class_ids, confidences

    def _filter_and_process(self, boxes, confidences, class_ids):
        filtered = []
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            class_name = self.class_names[class_id]
            if confidence > self.conf and class_name in self.interest_classes:
                filtered.append(([int(c) for c in box], float(confidence), class_name))
        return self._process_filtered(filtered)

    def _process_filtered(self, filtered):
        if not filtered:
            return []
        boxes, confidences, classes = zip(*filtered)
        return self._nms(boxes, confidences, classes)


# Working only on batch size 1
class AdapterYOLOX(AdapterOpenCV):
    """
    Adapter for YOLOX models with batch processing support.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = (416, 416)

    def __demo_postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            grid = np.stack(np.meshgrid(np.arange(wsize), np.arange(hsize)), 2)
            grid = grid.reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, axis=1)
        expanded_strides = np.concatenate(expanded_strides, axis=1)

        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        return outputs

    def post_processing(self, outputs: list, image_sizes: list, **kwargs):
        batch_detections = []
        input_size = (416, 416)

        for output, (img_w, img_h) in zip(outputs, image_sizes):
            if output.ndim == 3 and output.shape[0] == 1:
                output = output[0]  # (N, 85)

            output = self.__demo_postprocess(output, input_size)

            bboxes = output[:, :4]
            scores = output[:, 4:5] * output[:, 5:]
            class_ids = scores.argmax(axis=1)
            confidences = scores[np.arange(len(scores)), class_ids]

            detections = self._process_bboxes(
                bboxes=bboxes,
                class_ids=class_ids,
                confidences=confidences,
                image_dimensions=(img_w, img_h)
            )
            batch_detections.append(detections)

        return batch_detections

    def _process_bboxes(self, bboxes, class_ids, confidences, image_dimensions):
        final_boxes = []
        final_classes = []
        final_confidences = []
        img_w, img_h = image_dimensions  # Unpack image dimensions

        for bbox, class_id, confidence in zip(bboxes, class_ids, confidences):
            class_name = self.class_names[class_id]
            if confidence < self.conf or class_name not in self.interest_classes:
                continue

            coordinates = self._calculate_coordinates(bbox, img_w, img_h)
            final_boxes.append(coordinates)
            final_classes.append(class_name)
            final_confidences.append(float(confidence))

        return self._nms(final_boxes, final_confidences, final_classes)

    def _calculate_coordinates(self, bbox, img_w, img_h):
        input_w, input_h = self.input_size
        x_center, y_center, width, height = bbox
        x0 = int((x_center - width / 2) / (input_w / img_w))
        y0 = int((y_center - height / 2) / (input_h / img_h))
        x1 = int((x_center + width / 2) / (input_w / img_w))
        y1 = int((y_center + height / 2) / (input_h / img_h))
        return x0, y0, x1, y1


class AdapterUltralytics(Adapter):
    """
    Adapter for YOLO and RT-DETR using the Ultralytics ONNX runtime with batch support.
    """
    def pre_processing(self, images: list, **kwargs):
        return images

    def post_processing(self, outputs: list, image_sizes: list, **kwargs):
        batch_detections = []
        for result in outputs:
            detections = self._process_result(result)
            detections = self.__apply_nms(detections)
            batch_detections.append(detections)
        return batch_detections

    def _process_result(self, result):
        boxes = result.boxes
        detections = []
        for box, conf, cls_id in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy()
        ):
            class_name = self.class_names[int(cls_id)]
            if conf < self.conf:
                continue
            if self.interest_classes and class_name not in self.interest_classes:
                continue

            x0, y0, x1, y1 = map(int, box)
            detections.append([class_name, x0, y0, x1, y1, float(conf)])
        return detections

    def __apply_nms(self, detections: list):
        if not detections:
            return []

        boxes = np.array([det[1:5] for det in detections])
        confidences = np.array([det[5] for det in detections])
        indexes = cv.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), self.conf, self.nms)

        if len(indexes) == 0:
            return []

        indexes = indexes.flatten()
        return [detections[i] for i in indexes]


class AdapterSSDLite(Adapter):
    """
    Adapter for SSDLite320_MobileNet_V3_Large with reduced local variables.
    """
    def pre_processing(self, images: list[np.ndarray], **kwargs):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((320, 320)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
        return torch.stack([transform(img) for img in images])

    def post_processing(self, outputs: list, image_sizes: list, **kwargs):
        batch_detections = []
        for output, img_dim in zip(outputs, image_sizes):
            detections = self._process_output(output, img_dim)
            batch_detections.append(self.__apply_nms(detections))
        return batch_detections

    def _process_output(self, output, img_dim):
        detections = []
        for box, score, label in zip(output['boxes'],
                                     output['scores'],
                                     output['labels']):
            detection = self._create_detection(box, score, label, img_dim)
            if detection:
                detections.append(detection)
        return detections

    def _create_detection(self, box, score, label, image_dimensions):
        if score.item() < self.conf:
            return None

        class_id = label.item()
        if class_id >= len(self.class_names):
            return None

        class_name = self.class_names[class_id - 1]
        if self.interest_classes and class_name not in self.interest_classes:
            return None

        w, h = image_dimensions
        x1, y1, x2, y2 = self._scale_coordinates(box, w, h)
        return [class_name, x1, y1, x2, y2, float(score.item())]

    def _scale_coordinates(self, box, w, h):
        x1, y1, x2, y2 = box.tolist()
        scale_x, scale_y = w / 320, h / 320
        return (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        )

    def __apply_nms(self, detections: list):
        if not detections:
            return []

        boxes = np.array([[x, y, x2 - x, y2 - y] for _, x, y, x2, y2, _ in detections])
        confidences = np.array([conf for *_, conf in detections])

        indexes = cv.dnn.NMSBoxes(
            boxes.tolist(),
            confidences.tolist(),
            self.conf,
            self.nms
        )

        return [detections[i] for i in indexes.flatten()] if len(indexes) > 0 else []


class AdapterYOLOv4(Adapter):
    def pre_processing(self, images: np.ndarray, **kwargs):
        res_images = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            ih, iw = kwargs['size']
            h, w, _ = image.shape

            scale = min(iw / w, ih / h)
            nw, nh = int(scale * w), int(scale * h)
            image_resized = cv.resize(image, (nw, nh))

            image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
            dw, dh = (iw - nw) // 2, (ih - nh) // 2
            image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized
            image_padded = image_padded / 255.

            image_padded = image_padded.astype(np.float32)
            res_images.append(image_padded)

        result = np.stack(res_images, axis=0)
        return result

    def post_processing(self, outputs: list, image_sizes: list, **kwargs):
        # ANCHORS = "./models/yolov4_anchors.txt"
        ANCHORS = kwargs["path_anchors"]
        STRIDES = [8, 16, 32]
        XYSCALE = [1.2, 1.1, 1.05]
        input_size = 416

        ANCHORS = self.get_anchors(ANCHORS)
        STRIDES = np.array(STRIDES)

        n = outputs[0].shape[0]
        result = []
        for i in range(n):
            batch_i = [tensor[i][np.newaxis, ...] for tensor in outputs]
            pred_bbox = self.postprocess_bbbox(batch_i, ANCHORS, STRIDES, XYSCALE)
            bboxes = self.postprocess_boxes(pred_bbox, image_sizes[i][::-1], input_size, 0.25)
            bboxes = self._nms(bboxes, self.nms)

            result.append(self.format_data(bboxes))

        return result

    def format_data(self, detections):
        result = []
        for _, det in enumerate(detections):
            class_name = self.class_names[int(det[5])]
            result.append([class_name, *list(map(int, det[:4])), float(det[4])])

        return result

    @staticmethod
    def get_anchors(anchors_path, tiny=False):
        '''loads the anchors from a file'''
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = np.array(anchors.split(','), dtype=np.float32)
        return anchors.reshape(3, 3, 2)

    @staticmethod
    def postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE=(1, 1, 1)):
        '''define anchor boxes'''
        for i, pred in enumerate(pred_bbox):
            conv_shape = pred.shape
            output_size = conv_shape[1]
            conv_raw_dxdy = pred[:, :, :, :, 0:2]
            conv_raw_dwdh = pred[:, :, :, :, 2:4]
            xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
            xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)

            xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
            xy_grid = xy_grid.astype(np.float64)

            pred_xy = ((special.expit(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
            pred_wh = (np.exp(conv_raw_dwdh) * ANCHORS[i])
            pred[:, :, :, :, 0:4] = np.concatenate([pred_xy, pred_wh], axis=-1)

        pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = np.concatenate(pred_bbox, axis=0)
        return pred_bbox

    @staticmethod
    def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
        '''remove boundary boxs with a low detection probability'''
        valid_scale = [0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        org_h, org_w = org_img_shape
        resize_ratio = min(input_size / org_w, input_size / org_h)

        dw = (input_size - resize_ratio * org_w) / 2
        dh = (input_size - resize_ratio * org_h) / 2

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # # (3) clip some boxes that are out of range
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # # (4) discard some invalid boxes
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # # (5) discard some boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    @staticmethod
    def bboxes_iou(boxes1, boxes2):
        """calculate the Intersection Over Union value"""
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        return ious

    def _nms(self, bboxes, iou_threshold, sigma=0.3, method='nms'):
        """
        :param bboxes: (xmin, ymin, xmax, ymax, score, class)

        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
              https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = self.bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes

