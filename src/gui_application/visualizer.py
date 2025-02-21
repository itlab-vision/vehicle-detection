import cv2 as cv
from ..utils.frame_data_reader import FrameDataReader
from ..vehicle_detector.fake_detector import FakeDetector
class Visualize:
    def __init__(self, datareader:FrameDataReader, detector:FakeDetector, gt_data:list):
        self.datareader = datareader
        self.detector = detector
        self.gt_layout = gt_data

    def show(self):
        try:
            frame_idx = 0
            for image in self.datareader:
                if image is None:
                    break
                for box in self.detector.detect(image):
                    self.__draw_box(image, box, (255, 0, 0))
                if self.gt_layout:
                    for box in self.__get_groundtruth_bboxes(frame_idx):
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

    def __get_groundtruth_bboxes(self, frame_idx):
        return [item[1:] for item in self.gt_layout if item[0] == frame_idx]
