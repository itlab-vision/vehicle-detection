from pathlib import Path
import cv2 as cv
import numpy as np
import random
from abc import ABC, abstractmethod
import adapter

class Detector(ABC):
    
    @abstractmethod
    def detect(self, image):
        pass
    
    @staticmethod
    def create(model, path_classes, path_weights, path_config, conf, nms, scale, size, mean, swapRB):
        
        path_classes = Path(path_classes).absolute()
        if path_classes.exists():
            with open(path_classes, 'r') as f:
                class_names = f.read().split('\n')
        else:
            raise ValueError('Incorrect path to image.')
                
        if model == 'MobileNet':
            return VehicleDetectorMobileNet(path_weights, path_config, scale, size, mean, swapRB, adapter.AdapterMobileNet(conf, nms, class_names))
        elif model == 'YOLOv4':
            return VehicleDetectorYOLO(path_weights, path_config, scale, size, mean, swapRB, adapter.AdapterYOLO(conf, nms, class_names))
        elif model == 'YOLOv3_tiny':
            return VehicleDetectorYOLOv3Tiny(path_weights, path_config, scale, size, mean, swapRB, adapter.AdapterYOLOTiny(conf, nms, class_names))
        elif model == 'rcnn_resnet50':
            return VehicleDetectorRcnnResnet50(path_weights, path_config, scale, size, mean, swapRB, adapter.AdapterMaskRcnnResnet(conf, nms, class_names))
        elif model == 'rcnn_resnet_v2':
            return VehicleDetectorRcnnResnetV2(path_weights, path_config, scale, size, mean, swapRB, adapter.AdapterMaskRcnnResnet(conf, nms, class_names))
        elif model == "fake":
            return FakeDetector()
        else:
            raise ValueError(f"Unsupported model: {model}")

class VehicleDetectorRcnnResnet50(Detector):
    def __init__(self, path_weights, path_config, scale, size, mean, swapRB, adapter):
        
        self.scale = scale
        self.size = size
        self.mean = mean
        self.swapRB = swapRB
        self.adapter = adapter
        self.model = cv.dnn.readNet(model = path_weights, config = path_config, framework = 'TensorFlow')
    
    def detect(self, image):
        
        image_height, image_width, _ = image.shape
        blob = cv.dnn.blobFromImage(image = image, size = self.size, swapRB = self.swapRB)
        
        #blob = blob.swapaxes(1, 3)
        self.model.setInput(blob)
        #print(self.model.getLayerNames())
        
        boxes, masks = self.model.forward(['detection_out_final', 'detection_masks'])
        
        return self.adapter.postProcessing(boxes, image_width, image_height)
    
class VehicleDetectorRcnnResnetV2(Detector):
    def __init__(self, path_weights, path_config, scale, size, mean, swapRB, adapter):
        
        self.scale = scale
        self.size = size
        self.mean = mean
        self.swapRB = swapRB
        self.adapter = adapter
        self.model = cv.dnn.readNet(model = path_weights, config = path_config, framework = 'TensorFlow')
    
    def detect(self, image):
        
        image_height, image_width, _ = image.shape
        blob = cv.dnn.blobFromImage(image = image, size = self.size, swapRB = self.swapRB)

        self.model.setInput(blob)
        
        boxes = self.model.forward()
        
        return self.adapter.postProcessing(boxes, image_width, image_height)

class VehicleDetectorYOLO(Detector):
    
    def __init__(self, path_weights, path_config, scale, size, mean, swapRB, adapter):
        
        self.scale = scale
        self.size = size
        self.mean = mean
        self.swapRB = swapRB
        self.adapter = adapter
        self.model = cv.dnn.readNet(model = path_weights, config = path_config, framework = 'Darknet')
    
    def detect(self, image):
        
        image_height, image_width, _ = image.shape
        blob = cv.dnn.blobFromImage(image=image, scalefactor=self.scale, size=self.size, mean=self.mean, swapRB = self.swapRB)
        self.model.setInput(blob)
        output = self.model.forward()
        
        return self.adapter.postProcessing(output, image_width, image_height)
    
class VehicleDetectorYOLOv3Tiny(Detector):
    
    def __init__(self, path_weights, path_config, scale, size, mean, swapRB, adapter):
        
        self.scale = scale
        self.size = size
        self.mean = mean
        self.swapRB = swapRB
        self.adapter = adapter
        path_weights = Path(path_weights)
        self.model = cv.dnn.readNetFromONNX(path_weights.absolute())
    
    def detect(self, image):
        
        image_height, image_width, _ = image.shape
        blob = cv.dnn.blobFromImage(image=image, scalefactor=self.scale, size=self.size, mean=self.mean, swapRB = self.swapRB)
        self.model.setInput(blob)
        output = self.model.forward()
        
        return self.adapter.postProcessing(output, image_width, image_height)

class VehicleDetectorMobileNet(Detector):
   
    def __init__(self, path_weights, path_config, scale, size, mean, swapRB, adapter):
        
        self.scale = scale
        self.size = size
        self.mean = mean
        self.swapRB = swapRB
        self.adapter = adapter
        self.model = cv.dnn.readNet(model = path_weights, config = path_config, framework = 'TensorFlow')

    def detect(self, image):
        
        image_height, image_width, _ = image.shape
        blob = cv.dnn.blobFromImage(image = image, size = self.size, mean = self.mean, swapRB = self.swapRB)
        
        self.model.setInput(blob)
        output = self.model.forward()
        
        return self.adapter.postProcessing(output, image_width, image_height)
    
class FakeDetector(Detector):
    def __init__(self, seed = None):
        if seed is not None:
            random.seed(seed)

    @staticmethod
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
            
            bboxes.append((cl, x1, y1, x2, y2, 0.5))
        
        return bboxes