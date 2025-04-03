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
        self.scale = param_detect[0]
        self.size = param_detect[1]
        self.mean = param_detect[2]
        self.swapRB = param_detect[3]
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
                                         ad.AdapterYOLO(param_adapter[0],
                                         param_adapter[1], class_names))
        if adapter_name == 'AdapterYOLOTiny':
            return VehicleDetectorOpenCV('ONNX', paths, param_detect,
                                         ad.AdapterYOLOTiny(param_adapter[0],
                                         param_adapter[1], class_names))
        if adapter_name == 'AdapterDetectionTask':
            return VehicleDetectorOpenCV('TensorFlow', paths, param_detect,
                                         ad.AdapterDetectionTask(param_adapter[0],
                                         param_adapter[1], class_names))
        if adapter_name == 'AdapterFasterRCNN':
            return VehicleDetectorFasterRCNN(param_detect, 
                                             ad.AdapterFasterRCNN(param_adapter[0],
                                             param_adapter[1], class_names))
        if adapter_name == "fake":
            return FakeDetector()
        raise ValueError(f"Unsupported adapter: {adapter_name}")

class VehicleDetectorOpenCV(Detector):
    def __init__(self, format_load, paths, param_detect, adapter):
         
        super().__init__(param_detect, adapter)
        if format_load == 'TensorFlow':
            self.model = cv.dnn.readNetFromTensorflow(paths[0], paths[1])
        elif format_load == 'Darknet':
            self.model = cv.dnn.readNetFromDarknet(paths[1], paths[0])
        elif format_load == 'ONNX':
            self.model = cv.dnn.readNetFromONNX(paths[0])
        else:
            raise ValueError('Incorrect format load.')

    def detect(self, image):
        
        image_height, image_width, _ = image.shape
        blob = cv.dnn.blobFromImage(image=image, scalefactor=self.scale, size=self.size, mean=self.mean, swapRB = self.swapRB)
         
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
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
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

    def __init__(self, seed: int = None):
        """
        :param seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        super().__init__(None, None)

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
    