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
from pathlib import Path
from abc import ABC, abstractmethod
import random
import cv2 as cv
import numpy as np
import torch
import torchvision
import adapter as ad

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
    def detect(self, image: np.ndarray):
        """
        Process image and return detected objects.

        :param image: Input image array (OpenCV format)
        :return list: Detection tuples (label, x1, y1, x2, y2)
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
            with open(path_classes, 'r', encoding = 'utf-8') as f:
                class_names = f.read().split('\n')
        else:
            raise ValueError('Incorrect path to image.')

        if adapter_name == 'AdapterYOLO':
            return VehicleDetectorOpenCV('Darknet', paths, param_detect,
                                         ad.AdapterYOLO(param_adapter['confidence'],
                                         param_adapter['nms_threshold'], class_names))
        if adapter_name == 'AdapterYOLOTiny':
            return VehicleDetectorOpenCV('ONNX', paths, param_detect,
                                         ad.AdapterYOLOTiny(param_adapter['confidence'],
                                         param_adapter['nms_threshold'], class_names))
        if adapter_name == 'AdapterDetectionTask':
            return VehicleDetectorOpenCV('TensorFlow', paths, param_detect,
                                         ad.AdapterDetectionTask(param_adapter['confidence'],
                                         param_adapter['nms_threshold'], class_names))
        if adapter_name == 'AdapterFasterRCNN':
            return VehicleDetectorFasterRCNN(param_detect,
                                             ad.AdapterFasterRCNN(param_adapter['confidence'],
                                             param_adapter['nms_threshold'], class_names))
        if adapter_name == "fake":
            return FakeDetector()
        raise ValueError(f"Unsupported adapter: {adapter_name}")

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

    def detect(self, image):

        image_height, image_width, _ = image.shape
        blob = cv.dnn.blobFromImage(image=image, scalefactor=self.scale, size=self.size,
                                    mean=self.mean, swapRB = self.swap_rb)

        self.model.setInput(blob)
        boxes = self.model.forward()

        return self.adapter.post_processing(boxes, image_width, image_height)

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
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        self.model.eval()# Set the model to evaluation mode

    def detect(self, image: np.ndarray):
        """
        Performs object detection on the input image.

        :param image: Input image as a NumPy array.
        :return: List of detections in the format [class, x1, y1, x2, y2, confidence].
        """
        # Convert the image to RGB and preprocess it
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_tensor = torchvision.transforms.functional.to_tensor(image_rgb).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Post-process the detections using the adapter
        image_height, image_width, _ = image.shape
        return self.adapter.post_processing(outputs, image_width, image_height)

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

    def detect(self, image: list[np.ndarray]):
        """
        Process batch of images with simulated detection pipeline.
        
        :param image: List of numpy arrays (HWC images)
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
        for frame in image:
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
                    x1 = self.rand.randint(0, width-2)
                    x2 = self.rand.randint(x1+1, width-1)
                    y1 = self.rand.randint(0, height-2)
                    y2 = self.rand.randint(y1+1, height-1)
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
