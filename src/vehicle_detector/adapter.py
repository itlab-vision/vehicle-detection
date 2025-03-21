import numpy as np
import cv2 as cv
from abc import ABC, abstractmethod

class Adapter(ABC):
    
    @abstractmethod
    def postProcessing(self, output, image_width, image_height):
        pass

class AdapterMobileNet(Adapter):
    
    def __init__(self, conf, nms, class_names, interest_classes = ['car', 'bus', 'truck']):
        self.conf = conf
        self.nms = nms
        self.class_names = class_names
        self.interest_classes = interest_classes

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
                
                for i in range(len(self.interest_classes)):
                    if class_name == self.interest_classes[i]:
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

class AdapterMaskRcnnResnet(Adapter):
    
    def __init__(self, conf, nms, class_names, interest_classes = ['car', 'bus', 'truck']):
        self.conf = conf
        self.nms = nms
        self.class_names = class_names
        self.interest_classes = interest_classes

    def postProcessing(self, output, image_width, image_height):
        
        classes_id = []
        confidences = []
        boxes = []
        numDetections = output.shape[2]
        
        for i in range(numDetections):
            box = output[0, 0, i]
            confidence = box[2]
            if confidence > self.conf:
                class_id = int(box[1])
               
                left = min(int(box[3] * image_width), image_width)
                top = min(int(box[4] * image_height), image_height)
                right = min(int(box[5] * image_width), image_width)
                bottom = min(int(box[6] * image_height), image_height)
                
                class_name = self.class_names[int(class_id)]
                
                for i in range(len(self.interest_classes)):
                    if class_name == self.interest_classes[i]:
                        boxes.append((left, top, right - left, bottom - top))
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
    
    def __init__(self, conf, nms, class_names, interest_classes = ['car', 'bus', 'truck']):
        self.conf = conf
        self.nms = nms
        self.class_names = class_names
        self.interest_classes = interest_classes
    
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

                cx1 = int(detection[0] * image_width)
                cy1 = int(detection[1] * image_height)

                w = int(detection[2] * image_width)
                h = int(detection[3] * image_height)          

                for i in range(len(self.interest_classes)):
                    if class_name == self.interest_classes[i]:
                        boxes.append((cx1 - w // 2, cy1 - h // 2, w, h))
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

class AdapterYOLOTiny(Adapter):
    
    def __init__(self, conf, nms, class_names, interest_classes = ['car', 'bus', 'truck']):
        self.conf = conf
        self.tnms = nms
        self.class_names = class_names
        self.interest_classes = interest_classes
        
    def nms(self, boxes, scores, nms_thr):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets

    def demo_postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        return outputs

    def postProcessing(self, output, image_width, image_height):
        classes_id = []
        boxes = []
        confidences = []
        predictions = self.demo_postprocess(output[0], (416, 416))
        rh = 416 / image_height
        rw = 416 / image_width
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] / rw - boxes[:, 2]/2. / rw
        boxes_xyxy[:, 1] = boxes[:, 1] / rh - boxes[:, 3]/2. / rh
        boxes_xyxy[:, 2] = boxes[:, 0] / rw + boxes[:, 2]/2. / rw
        boxes_xyxy[:, 3] = boxes[:, 1] / rh + boxes[:, 3]/2. / rh
        dets = self.multiclass_nms_class_agnostic(boxes_xyxy, scores, nms_thr = self.tnms, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            bboxes = []
            for i in range(final_cls_inds.shape[0]):
                if final_scores[i] > self.conf:

                    class_id = int(final_cls_inds[i])
                
                    class_name = self.class_names[class_id]
                
                    x1 = int(final_boxes[i][0])
                    y1 = int(final_boxes[i][1])
                
                    x2 = int(final_boxes[i][2])
                    y2 = int(final_boxes[i][3])
                    for j in range(len(self.interest_classes)):
                        if class_name == self.interest_classes[j]:
                            bboxes.append((class_name, x1, y1, x2 , y2, final_scores[i]))

            return bboxes  