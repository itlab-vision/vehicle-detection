"""
Object Detection Module

Provides abstract detection interface and concrete implementations for:
- Production-ready vehicle detection (placeholder)
- Randomized fake detection for testing/development

Classes:
    :Detector: Abstract base class defining detection interface
    :VehicleDetector: (To be implemented) For real vehicle detection
    :FakeDetector: Test implementation with random bounding box generation

Dependencies:
    :OpenCV (cv2): for image handling
    :random: for synthetic detection generation
    :abc: for abstract base class support
"""

import random
from abc import ABC, abstractmethod
import numpy


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
    def detect(self, image: numpy.ndarray):
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

    def detect(self, image: numpy.ndarray):
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
