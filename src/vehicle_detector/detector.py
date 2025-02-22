"""
This module provides an abstract base class and concrete implementations for object detection,
with a focus on vehicle detection. It supports both real (to be implemented) and fake (randomized)
detection modes for testing and development purposes.

Classes
-------
Detector (Abstract Base Class)
VehicleDetector (Concrete Class)
FakeDetector (Concrete Class)
"""

import random
from abc import ABC, abstractmethod
import cv2 as cv


class Detector(ABC):
    @abstractmethod
    def detect(self, image):
        pass
    @staticmethod
    def create(mode):
        if mode == "vehicle":
            return VehicleDetector()
        if mode == "fake":
            return FakeDetector()
        raise ValueError(f"Unsupported mode: {mode}")

class VehicleDetector(Detector):
    def __init__(self):
        pass
    def detect(self, image):
        pass
class FakeDetector(Detector):
    def __init__(self, seed = None):
        if seed is not None:
            random.seed(seed)

    def detect(self, image):
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        bboxes = []
        num_boxes = random.randint(0, 7)
        for _ in range(num_boxes):
            if width < 2 or height < 2:
                continue
            cl = random.choice(["car", "bus", "truck"])
            x1 = random.randint(0, width - 2)
            x2 = random.randint(x1 + 1, width - 1)
            y1 = random.randint(0, height - 2)
            y2 = random.randint(y1 + 1, height - 1)
            bboxes.append((cl, x1, y1, x2, y2))
        return bboxes
