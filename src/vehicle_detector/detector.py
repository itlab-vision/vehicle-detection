import cv2 as cv
from cv2.dnn import DNN_BACKEND_CUDA, DNN_BACKEND_OPENCV
import numpy as np
import random
from abc import ABC, abstractmethod

class Adapter(ABC):
    
    @abstractmethod
    def postProcessing(self, output, image_width, image_height):
        pass

class AdapterMobileNet(Adapter):
    
    def __init__(self, conf, nms, class_names):
        self.conf = conf
        self.nms = nms
        self.class_names = class_names

    def postProcessing(self, output, image_width, image_height):
        classes_id = []
        boxes = []
        confidences = []
        for detection in output[0, 0, :, :]:
            confidence = detection[2]

            if confidence > self.conf:

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
                   
        indexes = cv.dnn.NMSBoxes(boxes, confidences, self.conf, self.nms)
        bboxes = []  
        for i in indexes:
            box = boxes[i]
            x1 = box[0]
            y1 = box[1]
            w = box[2]
            h = box[3]
            bboxes.append((classes_id[i], x1, y1, x1 + w, y1 + h, confidences[i]))
            
        return bboxes

class AdapterYOLO(Adapter):
    
    def __init__(self, conf, nms, class_names):
        self.conf = conf
        self.nms = nms
        self.class_names = class_names
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def postProcessing(self, output, image_width, image_height):
        classes_id = []
        boxes = []
        confidences = []
        for detection in output:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            class_name = self.class_names[class_id]
            if confidence > self.conf:

                сx1 = int(detection[0] * image_width)
                cy1 = int(detection[1] * image_height)

                w = int(detection[2] * image_width)
                h = int(detection[3] * image_height)          

                if class_name == 'car' or class_name == 'bus' or class_name == 'truck':
                    boxes.append((сx1 - w // 2, cy1 - h // 2, w, h))
                    classes_id.append(class_name)
                    confidences.append(confidence)

        indexes = cv.dnn.NMSBoxes(boxes, confidences, self.conf, self.nms)
        bboxes = []  
        for i in indexes:
            box = boxes[i]
            x1 = box[0]
            y1 = box[1]
            w = box[2]
            h = box[3]
            bboxes.append((classes_id[i], x1, y1, x1 + w, y1 + h, confidences[i]))

        return bboxes
    

        # classes_id = []
        # boxes = []
        # confidences = []
        # conf = 0.3
        
        # masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
        #        [59, 119], [116, 90], [156, 198], [373, 326]]
        
        # i = 0
        # for out in output:
            
        #     mask = masks[i % 3]
        #     i = i + 1
        #     box_confidence = self.sigmoid(out[..., 4])
        #     box_confidence = np.expand_dims(box_confidence, axis=-1)
        #     box_class_probs = self.sigmoid(out[..., 5:])

        #     box_scores = box_confidence * box_class_probs
        #     class_id = np.argmax(box_scores, axis=-1)
        #     confidence = np.max(box_scores, axis=-1)
        #     class_name = self.class_names[class_id]
        #     if confidence > conf:
                
               

        #         if class_name == 'car' or class_name == 'bus' or class_name == 'truck':
        #             # boxes.append((сx1 - w // 2, cy1 - h // 2, w, h))
        #             # classes_id.append(class_name)
        #             # confidences.append(confidence)
                    
        #             grid_h = 1
        #             grid_w = 1 

        #             anchors = [anchors[i] for i in mask]
        #             anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)

        #             # Reshape to batch, height, width, num_anchors, box_params.
        #             box_xy = self.sigmoid(out[..., :2])
        #             box_wh = np.exp(out[..., 2:4])
        #             box_wh = box_wh * anchors_tensor

                    

        #             col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        #             row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

        #             col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        #             row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        #             grid = np.concatenate((col, row), axis=-1)

        #             box_xy += grid
        #             box_xy /= (grid_w, grid_h)
        #             box_wh /= (416, 416)
        #             box_xy -= (box_wh / 2.)
        #             boxes = np.concatenate((box_xy, box_wh), axis=-1)
                 
        # indexes = cv.dnn.NMSBoxes(boxes, confidences, conf, 0.4)
        # bboxes = []  
        # for i in indexes:
        #     box = boxes[i]
        #     x1 = box[0]
        #     y1 = box[1]
        #     w = box[2]
        #     h = box[3]
        #     bboxes.append((classes_id[i], x1, y1, x1 + w, y1 + h, confidences[i]))
            
        # return bboxes
    

class Detector(ABC):
    
    @abstractmethod
    def detect(self, image):
        pass
    
    @staticmethod
    def create(model, path_classes, path_weights, path_config, conf, nms, mean):
        if model == 'MobileNet':
            with open(path_classes, 'r') as f:
                class_names = f.read().split('\n')
            return VehicleDetectorMobileNet(path_weights, path_config, mean, AdapterMobileNet(conf, nms, class_names))
        
        elif model == 'YOLOv3-tiny':
            with open(path_classes, 'r') as f:
                class_names = f.read().split('\n')
            return VehicleDetectorYOLO(path_weights, path_config, mean, AdapterYOLO(conf, nms, class_names))
        
        elif model == "fake":
            return FakeDetector()
        
        else:
            raise ValueError(f"Unsupported model: {model}")

class VehicleDetectorYOLO(Detector):
    
    def __init__(self, path_weights, path_config, mean, adapter):
        
        self.mean = mean
        self.adapter = adapter
        self.model = cv.dnn.readNet(model = self.path_weights, config = self.path_config, framework = 'Darknet')
        #self.model.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA) 
        #self.model.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    
    def detect(self, image):
        
        image_height, image_width, _ = image.shape
        blob = cv.dnn.blobFromImage(image=image, scalefactor=(1.0 / 255.0), size=(416, 416), mean=self.mean, swapRB = True, crop=False)
        self.model.setInput(blob)
        output = self.model.forward()
        
        return self.adapter.postProcessing(output, image_width, image_height)

class VehicleDetectorMobileNet(Detector):
   
    def __init__(self, path_weights, path_config, mean, adapter):
        
        self.mean = mean
        self.adapter = adapter
        self.model = cv.dnn.readNet(model = path_weights, config = path_config, framework = 'TensorFlow')
        #self.model.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA) 
        #self.model.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    def detect(self, image):
        
        image_height, image_width, _ = image.shape
        blob = cv.dnn.blobFromImage(image = image, size = (300, 300), mean = self.mean, swapRB=True)
        
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
            
            bboxes.append((cl, x1, y1, x2, y2))
        
        return []