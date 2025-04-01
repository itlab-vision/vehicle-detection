"""
Object Detection Module

Provides abstract detection interface and concrete implementations for:
- Real vehicle detection using Faster R-CNN (PyTorch-based)
- Fake detection with randomized bounding boxes for testing and development

Classes:
    :Detector: Abstract base class defining the detection interface
    :VehicleDetector: Placeholder for real vehicle detection system (currently not implemented)
    :FakeDetector: Generates random bounding boxes for testing purposes
    :VehicleDetectorFasterRCNN: Concrete implementation of vehicle detection
                                using Faster R-CNN (PyTorch)

Dependencies:
    :torch: for Faster R-CNN model and deep learning functionalities
    :torchvision: for pre-trained Faster R-CNN model and image transformations
    :cv2: for image handling and transformations (OpenCV)
    :random: for synthetic detection generation (FakeDetector)
    :numpy: for numerical operations
    :abc: for abstract base class support
"""

import random
from abc import ABC, abstractmethod
import numpy as np
import cv2 as cv
import torch
import torchvision
from src.vehicle_detector.adapter import AdapterFasterRCNN


class Detector(ABC):
    """
    Abstract base class for object detection implementations.
    
    Defines interface for detection systems using factory pattern.
    
    Methods:
        detect: Abstract detection method to be implemented
        create: Factory method for instantiating concrete detectors
    
    Supported Modes:
        'vehicle': Production detector (not implemented)
        'fake': Testing detector with random boxes
    """

    @abstractmethod
    def detect(self, image: np.ndarray):
        """
        Process image and return detected objects.

        :param image: Input image array (OpenCV format)
        :return list: Detection tuples (label, x1, y1, x2, y2)
        """

    @staticmethod
    def create(mode: str):
        """
        Factory method for creating detector instances.
        
        :param mode: Detector variant selector
        :return Detector: Concrete subclass instance
        :raise ValueError: For unsupported mode specifications
        """
        if mode == "vehicle":
            return VehicleDetector()
        if mode == "fake":
            return FakeDetector()
        if mode == "FasterRCNN":
            return VehicleDetectorFasterRCNN()
        raise ValueError(f"Unsupported mode: {mode}")


class VehicleDetector(Detector):
    """
    Placeholder for real vehicle detection system.

    Currently, returns empty detections.
    """

    def __init__(self):
        pass

    def detect(self, image):
        return []


class FakeDetector(Detector):
    """
    Testing detector generating random bounding boxes.
    
    Generates 0-7 random vehicle boxes per image with:
    - Random vehicle classes (car/bus/truck)
    - Random valid positions within image bounds
    - Reproducible results via seed control
    """

    def __init__(self, seed: int = None):
        """
        :param seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

    def detect(self, image: np.ndarray):
        """
        Generate synthetic detections for testing.
        
        :param image: Input image array (checks size validity)
        :return list: Detection tuples (class, x1, y1, x2, y2, confidence)
        """
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        bboxes = []
        num_boxes = random.randint(0, 5)
        chance = random.random()
        if chance < 0.5:
            return []
        for _ in range(num_boxes):
            if width < 2 or height < 2:
                continue
            cl = random.choice(["car", "bus", "truck"])
            x1 = random.randint(0, width - 2)
            x2 = random.randint(x1 + 1, width - 1)
            y1 = random.randint(0, height - 2)
            y2 = random.randint(y1 + 1, height - 1)
            confidence = random.random()
            bboxes.append((cl, x1, y1, x2, y2, confidence))
        return bboxes


class VehicleDetectorFasterRCNN(Detector):
    """
    Vehicle detector based on Faster R-CNN using a pre-trained PyTorch model.
    """

    def __init__(self, class_names: list = ('car', 'bus', 'truck'), conf_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        Initializes the Faster R-CNN vehicle detector.

        :param class_names: List of class names to be detected (e.g., ['car', 'bus']).
        :param conf_threshold: Confidence threshold for detections.
        :param nms_threshold: Non-Maximum Suppression (NMS) threshold.
        """
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()  # Set the model to evaluation mode
        self.adapter = AdapterFasterRCNN(conf_threshold, nms_threshold, class_names)

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
