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
import cv2 as cv
import numpy as np
import random
from abc import ABC, abstractmethod
import adapter

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
    
    def __init__(self, model_name, scale, size, mean, swapRB, adapter):
        self.model_name = model_name
        self.scale = scale
        self.size = size
        self.mean = mean
        self.swapRB = swapRB
        self.adapter = adapter
        
    @abstractmethod
    def detect(self, image):
        """
        Process image and return detected objects.
        
        :param image: Input image array (OpenCV format)
        :return: list: Detection tuples (label, x1, y1, x2, y2)
        """ 
        
    
    @staticmethod
    def create(model_name, path_classes, path_weights, path_config, conf, nms, scale, size, mean, swapRB):
        """
        Factory method for creating detector instances.
        
        :param mode: Detector variant selector
        :return: Detector: Concrete subclass instance
        :raise: ValueError: For unsupported mode specifications
        """
        path_classes = Path(path_classes).absolute()
        if path_classes.exists():
            with open(path_classes, 'r', 'utf-8') as f:
                class_names = f.read().split('\n')
        else:
            raise ValueError('Incorrect path to image.')
                
        if model_name == 'YOLOv4':
            return VehicleDetectorOpenCV(model_name, 'Darknet', path_weights, path_config, scale, size, mean, swapRB, adapter.AdapterYOLO(conf, nms, class_names))
        elif model_name == 'YOLOv3_tiny':
            return VehicleDetectorOpenCV(model_name, 'ONNX', path_weights, path_config, scale, size, mean, swapRB, adapter.AdapterYOLOTiny(conf, nms, class_names))
        elif model_name == 'rcnn_resnet50' or model_name == 'rcnn_resnet_v2' or model_name == 'efficientdet_d1' or model_name == 'efficientdet_d0' or model_name == 'lite_mobilenet_v2' or model_name == 'MobileNet':
            return VehicleDetectorOpenCV(model_name, 'TensorFlow', path_weights, path_config, scale, size, mean, swapRB, adapter.AdapterDetectionTask(conf, nms, class_names))
        elif model_name == "fake":
            return FakeDetector()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

class VehicleDetectorOpenCV(Detector):
    def __init__(self, model_name, format_load, path_weights, path_config, scale, size, mean, swapRB, adapter):
         
        super().__init__(model_name, scale, size, mean, swapRB, adapter)
        if format_load == 'TensorFlow':
            self.model = cv.dnn.readNetFromTensorflow(path_weights, path_config)
        elif format_load == 'Darknet':
            self.model = cv.dnn.readNetFromDarknet(path_config, path_weights)
        elif format_load == 'ONNX':
            self.model = cv.dnn.readNetFromONNX(path_weights)
        else:
            raise ValueError('Incorrect format load.')

    def detect(self, image):
        
        image_height, image_width, _ = image.shape
        blob = cv.dnn.blobFromImage(image=image, scalefactor=self.scale, size=self.size, mean=self.mean, swapRB = self.swapRB)
         
        self.model.setInput(blob)
        boxes = self.model.forward()
        
        return self.adapter.postProcessing(boxes, image_width, image_height)
     
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
        super().__init__(0, 0, 0, 0, 0, 0)
        if seed is not None:
            random.seed(seed)

    def detect(self, image: np.ndarray):
        """
        Generate synthetic detections for testing.
        
        :param image: Input image array (checks size validity)
        :return: list: Detection tuples (class, x1, y1, x2, y2, confidence)
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