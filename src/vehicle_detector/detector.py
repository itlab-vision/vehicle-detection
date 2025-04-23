"""
Object Detection Module

Provides abstract detection interface and concrete implementations for:
- Production-ready vehicle detection (placeholder)
- Randomized fake detection for testing/development

Classes:
    :Detector: Abstract base class defining detection interface
    :VehicleDetector: For real vehicle detection
    :FakeDetector: Test implementation with random bounding box generation

Dependencies:
    :OpenCV (cv2): for image handling
    :random: for synthetic detection generation
    :abc: for abstract base class support
"""
import time
from pathlib import Path
from abc import ABC, abstractmethod
import random
import cv2 as cv
import numpy as np
import torch
import torchvision
from torchvision.models import detection
import onnxruntime as rt
from ultralytics import YOLO, RTDETR

import src.vehicle_detector.adapter as ad


class Detector(ABC):
    """
    Abstract base class for object detection implementations.
    
    Defines interface for detection systems using factory pattern.
    
    Methods:
        detect: Abstract detection method
        create: Factory method for instantiating concrete detectors
    
    Supported Modes:
        'vehicle': Production detector
        'fake': Testing detector with random boxes
    """

    def __init__(self, param_detect, adapter):
        self.scale = param_detect['scale']
        self.size = param_detect['size']
        self.mean = param_detect['mean']
        self.swap_rb = param_detect['swapRB']
        self.adapter = adapter

    @abstractmethod
    def detect(self, images: list[np.ndarray]):
        """
        Process image and return detected objects.

        :param images: Input array of images (OpenCV format)
        """

    @staticmethod
    def create(adapter_name, path_classes, paths, param_adapter, param_detect):
        """
        Factory method for creating detector instances.
        
        :return: Detector: Concrete subclass instance
        :raise: ValueError: For unsupported mode specifications
        """
        path_classes = Path(path_classes).absolute()
        if path_classes.exists():
            with open(path_classes, 'r', encoding='utf-8') as f:
                class_names = f.read().split('\n')
        else:
            raise ValueError('Incorrect path to classes.')

        detector = None
        if adapter_name == 'AdapterYOLO':
            detector = VehicleDetectorOpenCV('Darknet', paths, param_detect,
                                         ad.AdapterYOLO(param_adapter['confidence'],
                                                        param_adapter['nms_threshold'],
                                                        class_names))
        elif adapter_name == 'AdapterYOLOTiny':
            detector = VehicleDetectorOpenCV('ONNX', paths, param_detect,
                                         ad.AdapterYOLOTiny(param_adapter['confidence'],
                                                            param_adapter['nms_threshold'],
                                                            class_names))
        elif adapter_name == 'AdapterYOLOX':
            detector = VehicleDetectorOpenCV('ONNX', paths, param_detect,
                                         ad.AdapterYOLOX(param_adapter['confidence'],
                                                         param_adapter['nms_threshold'],
                                                         class_names))
        elif adapter_name == 'AdapterDetectionTask':
            detector = VehicleDetectorOpenCV('TensorFlow', paths, param_detect,
                                         ad.AdapterDetectionTask(param_adapter['confidence'],
                                                                 param_adapter['nms_threshold'],
                                                                 class_names))
        elif adapter_name == 'AdapterFasterRCNN':
            detector = VehicleDetectorFasterRCNN(param_detect,
                                             ad.AdapterFasterRCNN(param_adapter['confidence'],
                                                                  param_adapter['nms_threshold'],
                                                                  class_names))
        elif adapter_name == 'AdapterUltralytics':
            detector = VehicleDetectorYoloUltralytics(paths, param_detect,
                                                      ad.AdapterUltralytics(param_adapter['confidence'],
                                                                                param_adapter['nms_threshold'],
                                                                                class_names))
        elif adapter_name == 'AdapterSSDLite':
            detector = VehicleDetectorSSDLite(param_detect,
                                              ad.AdapterSSDLite(param_adapter['confidence'],
                                                                param_adapter['nms_threshold'],
                                                                class_names))
        elif adapter_name == 'AdapterYOLOv4':
            detector = VehicleDetectorYOLOv4(paths, param_detect,
                                             ad.AdapterYOLOv4(param_adapter['confidence'],
                                                              param_adapter['nms_threshold'],
                                                              class_names))
        elif adapter_name == "fake":
            detector = FakeDetector()
        else:
            raise ValueError(f"Unsupported adapter: {adapter_name}")

        return detector


# need testing of batch processing implementation
class VehicleDetectorOpenCV(Detector):
    """
    vehicle detection
    """

    def __init__(self, format_load, paths, param_detect, adapter):

        super().__init__(param_detect, adapter)
        if format_load == 'TensorFlow':
            self.model = cv.dnn.readNetFromTensorflow(paths['path_weights'], paths['path_config'])
        elif format_load == 'Darknet':
            self.model = cv.dnn.readNetFromDarknet(paths['path_config'], paths['path_weights'])
        elif format_load == 'ONNX':
            self.model = cv.dnn.readNetFromONNX(paths['path_weights'])
        else:
            raise ValueError('Incorrect format load.')

    def detect(self, images: list[np.ndarray]):
        # Pre-process image using the adapter
        start_time = time.time()
        image_transformed = self.adapter.pre_processing(images,
                                                        scalefactor=self.scale,
                                                        size=self.size,
                                                        mean=self.mean,
                                                        swapRB=self.swap_rb)
        blobs = cv.dnn.blobFromImages(image_transformed)
        preproc_time = time.time() - start_time

        # Perform inference
        start_time = time.time()
        self.model.setInput(blobs)
        outputs = self.model.forward()
        inference_time = time.time() - start_time

        # Post-process the detections using the adapter
        start_time = time.time()
        image_sizes = [(img.shape[1], img.shape[0]) for img in images]  # (width, height)
        detections = self.adapter.post_processing(outputs, image_sizes)
        postproc_time = time.time() - start_time

        return detections, preproc_time, inference_time, postproc_time


class VehicleDetectorFasterRCNN(Detector):
    """
    Vehicle detector based on Faster R-CNN using a pre-trained PyTorch model.
    """

    def __init__(self, param_detect, adapter):
        """
        Initializes the Faster R-CNN vehicle detector.

        :param class_names: List of class names to be detected (e.g., ['car', 'bus']).
        :param conf_threshold: Confidence threshold for detections.
        :param nms_threshold: Non-Maximum Suppression (NMS) threshold.
        """
        super().__init__(param_detect, adapter)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        self.model.eval()  # Set the model to evaluation mode

    def detect(self, images: list[np.ndarray]):
        """
        Performs object detection on the input image.

        :param images: Input list of images (image is a NumPy array).
        :return: List of detections in the format [class, x1, y1, x2, y2, confidence].
        """
        # Pre-process
        start_time = time.time()
        image_tensors = self.adapter.pre_processing(images)
        preproc_time = time.time() - start_time

        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(image_tensors)
        inference_time = time.time() - start_time

        # post-processing
        start_time = time.time()
        image_sizes = [(img.shape[1], img.shape[0]) for img in images]  # (width, height)
        detections = self.adapter.post_processing(outputs, image_sizes)
        postproc_time = time.time() - start_time

        return detections, preproc_time, inference_time, postproc_time


class VehicleDetectorYoloUltralytics(Detector):
    """
    Vehicle detector based on YOLO and RTDETR ONNX using the Ultralytics API.
    """

    def __init__(self, paths, param_detect, adapter):
        """
        :param param_detect: Dictionary with detection parameters.
        :param adapter: Adapter for pre/post-processing.
        """
        super().__init__(param_detect, adapter)
        if 'yolo' in paths['path_weights']:
            self.model = YOLO(paths['path_weights'])
        elif 'rtdetr' in paths['path_weights']:
            self.model = RTDETR(paths['path_weights'])
        else:
            raise "VehicleDetectorYoloUltralytics invalid path_weights"

    def detect(self, images: list[np.ndarray]):
        """
        Performs object detection on the input images using YOLOv8.

        :param images: List of input images (BGR numpy arrays).
        :return: Tuple (detections, preproc_time, inference_time, postproc_time)
        """
        # Pre-processing
        start_time = time.time()
        preprocessed = self.adapter.pre_processing(images,
                                                  size=self.size)
        preproc_time = time.time() - start_time

        # Inference
        start_time = time.time()
        results = self.model(preprocessed)
        inference_time = time.time() - start_time

        # Post-processing
        start_time = time.time()
        image_sizes = [(img.shape[1], img.shape[0]) for img in images]
        detections = self.adapter.post_processing(results, image_sizes)
        postproc_time = time.time() - start_time

        return detections, preproc_time, inference_time, postproc_time


class VehicleDetectorSSDLite(Detector):
    """
    Vehicle detector using SSDLite320_MobileNet_V3_Large.
    """

    def __init__(self, param_detect, adapter):
        super().__init__(param_detect, adapter)
        self.model = detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        self.model.eval()

    def detect(self, images: list[np.ndarray]):
        start_time = time.time()
        preprocessed = self.adapter.pre_processing(images)
        preproc_time = time.time() - start_time

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(preprocessed)
        inference_time = time.time() - start_time

        start_time = time.time()
        image_sizes = [(img.shape[1], img.shape[0]) for img in images]
        detections = self.adapter.post_processing(outputs, image_sizes)
        postproc_time = time.time() - start_time

        return detections, preproc_time, inference_time, postproc_time


class VehicleDetectorYOLOv4(Detector):
    """
    Vehicle detector using YOLOv4
    """

    def __init__(self, paths, param_detect, adapter):
        super().__init__(param_detect, adapter)
        self.path_anchors = paths['path_anchors']
        self.model = rt.InferenceSession(paths['path_weights'])

    def detect(self, images: list[np.ndarray]):
        start_time = time.time()
        preprocessed = self.adapter.pre_processing(images,
                                                   size=self.size)
        preproc_time = time.time() - start_time

        start_time = time.time()
        outputs = self.model.get_outputs()
        output_names = list(map(lambda output: output.name, outputs))
        input_name = self.model.get_inputs()[0].name

        outputs = self.model.run(output_names, {input_name: preprocessed})
        inference_time = time.time() - start_time

        start_time = time.time()
        image_sizes = [(img.shape[1], img.shape[0]) for img in images]
        detections = self.adapter.post_processing(outputs, image_sizes,
                                                  path_anchors=self.path_anchors)
        postproc_time = time.time() - start_time

        return detections, preproc_time, inference_time, postproc_time


class FakeDetector(Detector):
    """
    Testing detector generating random bounding boxes.

    Generates 0-7 random vehicle boxes per image with:
    - Random vehicle classes (car/bus/truck)
    - Random valid positions within image bounds
    - Reproducible results via seed control
    """

    def __init__(self,
                 seed: int = None,
                 time_ranges: tuple[tuple[float, float]] = ((0.01, 0.05),
                                                            (0.08, 0.3),
                                                            (0.01, 0.1))):
        """
        Initialize fake detector with controlled randomness.

        :param seed: Random seed for reproducibility
        :param time_ranges: Tuple of (preproc, inference, postproc) time ranges
        """
        self.rand = random.Random(seed)
        self.time_ranges = time_ranges
        self.classes = ["car", "bus", "truck"]
        super().__init__(None, None)

    def detect(self, images: list[np.ndarray]):
        """
        Process batch of images with simulated detection pipeline.

        :param images: List of numpy arrays (HWC images)
        :return: Tuple containing:
            - List of detections per image
            - Preprocessing time
            - Inference time
            - Postprocessing time
        """
        preproc = self.rand.uniform(*self.time_ranges[0])
        inference = self.rand.uniform(*self.time_ranges[1])
        postproc = self.rand.uniform(*self.time_ranges[2])

        batch_detections = []
        for frame in images[0]:
            if frame is None or frame.size == 0:
                batch_detections.append([])
                continue

            height, width = frame.shape[:2]
            detections = []

            if self.rand.random() < 0.3:
                batch_detections.append([])
                continue

            for _ in range(self.rand.randint(0, 5)):
                try:
                    x1 = self.rand.randint(0, width - 2)
                    x2 = self.rand.randint(x1 + 1, width - 1)
                    y1 = self.rand.randint(0, height - 2)
                    y2 = self.rand.randint(y1 + 1, height - 1)
                except ValueError:
                    continue

                detections.append((
                    self.rand.choice(self.classes),
                    x1, y1, x2, y2,
                    round(self.rand.uniform(0.4, 0.99), 2)
                ))

            batch_detections.append(detections)

        return (
            batch_detections,
            preproc,
            inference,
            postproc
        )
