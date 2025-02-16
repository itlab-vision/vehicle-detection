import cv2 as cv
from datareader import DataReader
import random
class PseudoDetector:
    def __init__(self, annotation_file):
        self.frame_annotations = self.__rd_form_ret_annotations(annotation_file)
    def __rd_form_ret_annotations(self, annotation_file):
        rand_bias = self.__get_random_bias()
        frame_annotations = {}
        with open(annotation_file, 'r') as file:
            annotations = file.readlines()
        
        for line in annotations:
            parts = line.strip().split()
            frame_idx = int(parts[0])
            label = parts[1]
            x1, y1 = int(parts[2]) + rand_bias, int(parts[3]) - rand_bias
            x2, y2 = int(parts[4]) - rand_bias, int(parts[5]) + rand_bias
            
            if frame_idx not in frame_annotations:
                frame_annotations[frame_idx] = []
            frame_annotations[frame_idx].append((label, x1, y1, x2, y2))
        return frame_annotations

    def __get_random_bias(self):
        random.seed()
        return random.randint(10, 15)

    
    
    def get_annotations(self):
        return self.frame_annotations
    

class Visualize:
    def __init__(self, datareader:DataReader, detector:PseudoDetector):
        self.datareader = datareader
        self.detector = detector
    def show(self):
        try:
            groundtruth = self.__rd_form_ret_groundtruth()
            detector_boxes = self.detector.get_annotations()

            frame_idx = 0
            for image in self.datareader: # How to split?
                if image is None:
                    break
                
                for box in detector_boxes[frame_idx]:
                    self.__draw_box(image, box, (255, 0, 0))

                if groundtruth:
                    for box in groundtruth[frame_idx]:
                        self.__draw_box(image, box, (0, 255, 0))

                frame_idx+=1
                if cv.waitKey(25) & 0xFF == ord('q'):
                    break      
        except Exception as e:
            print(f'An error occurred: {e}')
        finally:
            cv.destroyAllWindows()

    def __draw_box(self, image, box, color):
        label, x1, y1, x2, y2 = box
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv.putText(image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.imshow('Image', image)
            
    def __rd_form_ret_groundtruth(self):
        groundtruth = {}
        with open(self.datareader.groundtruth_path, 'r') as file:
            annotations = file.readlines()
        
        for line in annotations:
            parts = line.strip().split()
            frame_idx = int(parts[0])
            label = parts[1]
            x1, y1 = int(parts[2]), int(parts[3])
            x2, y2 = int(parts[4]), int(parts[5])
            
            if frame_idx not in groundtruth:
                groundtruth[frame_idx] = []
            groundtruth[frame_idx].append((label, x1, y1, x2, y2))
        return groundtruth


