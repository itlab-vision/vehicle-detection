"""
Computer Vision Visualization Module

Provides functionality for visualizing object detection results alongside ground truth data.
Handles both image sequences and video inputs with the following components:
- Argument parsing for input configuration
- Frame data reading (images/video)
- Fake object detection implementation
- Visualization of detection vs groundtruth comparisons

Dependencies:
- OpenCV (cv2) for image processing and display
- argparse for command-line argument handling
"""

import cv2 as cv
from src.utils.frame_data_reader import FrameDataReader
from src.utils.writer import Writer
from src.vehicle_detector.detector import Detector


class Visualize:
    """
    Visualization controller for detection/groundtruth comparison.

    Handles frame iteration, bounding box drawing, and display management.
    Uses different colors for detected boxes (blue) and groundtruth boxes (green).
    """

    def __init__(self, datareader:FrameDataReader, writer:Writer, detector:Detector, gt_data:list):
        """
        Initialize visualization components with data sources.

        :param datareader: Input source for frames (video/images)
        :param detector: Detection component
        :param gt_data: Loaded groundtruth in format [[frame_idx, label, x1, y1, x2, y2], ...]
        """
        self.datareader = datareader
        self.writer = writer
        self.detector = detector
        self.gt_layout = gt_data

    def show(self):
        """
        Main visualization loop.
        
        Processes frames sequentially with the following workflow:
        
        1. Retrieves next frame from data reader
        2. Runs object detection
        3. Write retrieved data from detection if available
        4. Draws blue bounding boxes
        3. Overlays green groundtruth boxes if available
        4. Displays combined visualization
        5. Handles exit condition (Q key press)
        
        Performs cleanup on exit or error, closing all OpenCV windows.
        """
        try:
            frame_idx = 0
            for image in self.datareader:
                if image is None:
                    break
                for box in self.detector.detect(image):
                    self.__draw_box(image, box, (255, 112, 166))
                    if self.writer:
                        self.writer.write((frame_idx, *box))
                if self.gt_layout:
                    for box in self.__get_groundtruth_bboxes(frame_idx):
                        self.__draw_box(image, box, (0, 255, 0))
                frame_idx+=1
                if cv.waitKey(25) & 0xFF == ord('q'):
                    break
        except Exception as e:
            raise Exception(e)
        finally:
            if self.writer:
                self.writer.close()
            cv.destroyAllWindows()

    @staticmethod
    def __draw_box(image, box, color):
        """
        Internal method: Draw single bounding box with label.

        :param image: Input frame matrix
        :param box: Detection tuple (label, x1, y1, x2, y2, confidence(detector) )
        :param color: BGR tuple for box/label color
        """
        label, x1, y1, x2, y2 = box[:5]
        confidence = ''
        if len(box) == 6:
            confidence = str(box[5])
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv.putText(image, label, (x1, y1 + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.putText(image, confidence, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
        cv.imshow("Detection", image)

    def __get_groundtruth_bboxes(self, frame_idx):
        """
        Internal method: Filter groundtruth boxes for current frame.
        
        :param frame_idx: Current frame index
        :return:list: Groundtruth boxes for specified frame in format [label, x1, y1, x2, y2]
        """
        return [item[1:] for item in self.gt_layout if item[0] == frame_idx]
