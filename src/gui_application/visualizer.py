import cv2 as cv
from ..utils.data_reader import DataReader, GroundtruthReader, FakeGTReader
from ..vehicle_detector.fake_detector import FakeDetector
import random
# class PseudoDetector:
#     def __init__(self, annotation_file):
#         self.frame_annotations = self.__rd_form_ret_annotations(annotation_file)
#     def __rd_form_ret_annotations(self, annotation_file):
#         rand_bias = self.__get_random_bias()
#         frame_annotations = {}
#         with open(annotation_file, 'r') as file:
#             annotations = file.readlines()
        
#         for line in annotations:
#             parts = line.strip().split()
#             frame_idx = int(parts[0])
#             label = parts[1]
#             x1, y1 = int(parts[2]) + rand_bias, int(parts[3]) - rand_bias
#             x2, y2 = int(parts[4]) - rand_bias, int(parts[5]) + rand_bias
            
#             if frame_idx not in frame_annotations:
#                 frame_annotations[frame_idx] = []
#             frame_annotations[frame_idx].append((label, x1, y1, x2, y2))
#         return frame_annotations

#     def __get_random_bias(self):
#         random.seed()
#         return random.randint(10, 15)

    
    
#     def get_annotations(self):
#         return self.frame_annotations

class Visualize:
    def __init__(self, datareader:DataReader, detector:FakeDetector, gt_path:str):
        self.datareader = datareader
        self.detector = detector
        self.gt_data = FakeGTReader().read()
    def show(self):
        try:
            frame_idx = 0
            for image in self.datareader:
                if image is None:
                    break
                
                for box in self.detector.detect(image):
                    self.__draw_box(image, box, (255, 0, 0))

                if self.gt_data:
                    for box in self.__format_groundtruth(frame_idx):
                        self.__draw_box(image, box, (0, 255, 0))

                frame_idx+=1
                if cv.waitKey(25) & 0xFF == ord('q'):
                    break      
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            cv.destroyAllWindows()

    def __draw_box(self, image, box, color):
        label, x1, y1, x2, y2 = box
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv.putText(image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.imshow("Image", image)
    
    def __format_groundtruth(self, frame_idx):
        
        return [item[1:] for item in self.gt_data if item[0] == frame_idx]


