import cv2 as cv
import numpy as np
import random
from abc import ABC, abstractmethod

class Detector(ABC):
    
    @abstractmethod
    def detect(self, image):
        pass
    
    @staticmethod
    def create(model_path, model):
        if model == 'MobileNet':
            return VehicleDetector(model_path)
        elif model == 'YOLOv3-tiny':
            return VehicleDetectorYOLO(model_path)
        elif model == "fake":
            return FakeDetector()
        else:
            raise ValueError(f"Unsupported model: {model}")

class VehicleDetectorYOLO(Detector):
    
    def __init__(self, model_path):
        self.path_classes = model_path + "classes_coco.txt"
        self.path_weights = model_path + "yolov3-tiny.weights"
        self.path_config = model_path + "yolov3-tiny.cfg"
        with open(self.path_classes, 'r') as f:
            self.class_names = f.read().split('\n')

        self.model = cv.dnn.readNet(model = self.path_weights, config = self.path_config, framework = 'Darknet')
        
    def detect(self, image):
        
        image_height, image_width, _ = image.shape
        blob = cv.dnn.blobFromImage(image=image, scalefactor=(1.0 / 255.0), size=(416, 416), mean=(104, 117, 123), swapRB = True, crop=False)

        self.model.setInput(blob)
        output = self.model.forward()
        
        classes_id = []
        boxes = []
        confidences = []
        conf = 0.3
        for detection in output:
            
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            class_name = self.class_names[class_id]
            if confidence > conf:
                
                сx1 = int(detection[0] * image_width)
                cy1 = int(detection[1] * image_height)
                
                w = int(detection[2] * image_width)
                h = int(detection[3] * image_height)          

                if class_name == 'car' or class_name == 'bus' or class_name == 'truck':
                    boxes.append((сx1 - w // 2, cy1 - h // 2, w, h))
                    classes_id.append(class_name)
                    confidences.append(confidence)
                 
        indexes = cv.dnn.NMSBoxes(boxes, confidences, conf, 0.4)
        bboxes = []  
        for i in indexes:
            box = boxes[i]
            x1 = box[0]
            y1 = box[1]
            w = box[2]
            h = box[3]
            bboxes.append((classes_id[i], x1, y1, x1 + w, y1 + h, confidences[i]))
            
        return bboxes


class VehicleDetector(Detector):
   
    def __init__(self, model_path):
        
        self.path_classes = model_path + "classes_coco.txt"
        self.path_inference = model_path + "frozen_inference_graph.pb"
        self.path_config = model_path + "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
        with open(self.path_classes, 'r') as f:
            self.class_names = f.read().split('\n')

        self.model = cv.dnn.readNet(model = self.path_inference, config = self.path_config, framework = 'TensorFlow')
    
    def detect(self, image):
        
        image_height, image_width, _ = image.shape
        blob = cv.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)
        self.model.setInput(blob)
        output = self.model.forward()
        
        classes_id = []
        boxes = []
        confidences = []
        conf = 0.3
        for detection in output[0, 0, :, :]:
            confidence = detection[2]

            if confidence > conf:

                class_id = detection[1]
                
                class_name = self.class_names[int(class_id)-1]
                
                x1 = int(detection[3] * image_width)
                y1 = int(detection[4] * image_height)
                
                x2 = int(detection[5] * image_width)
                y2 = int(detection[6] * image_height)
                
                if class_name == 'car' or class_name == 'bus' or class_name == 'truck':
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
                    classes_id.append(class_name)
                    confidences.append(confidence)
                   
        indexes = cv.dnn.NMSBoxes(boxes, confidences, conf, 0.4)
        bboxes = []  
        for i in indexes:
            box = boxes[i]
            x1 = box[0]
            y1 = box[1]
            w = box[2]
            h = box[3]
            bboxes.append((classes_id[i], x1, y1, x1 + w, y1 + h, confidences[i]))
            
        return bboxes
    
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
            
            bboxes.append((cl, x1, y1, x2, y2))
        
        return []