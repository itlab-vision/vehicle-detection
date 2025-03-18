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
from tqdm import tqdm
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
        self.progress_bar = None

    def show(self):
        """
        Main visualization loop.
        
        Processes frames sequentially with the following workflow:
        
        1. Retrieves next frame from data reader
        2. Runs object detection
        3. Write retrieved data from detection and write in file if available
        4. Draws blue bounding boxes
        5. Overlays green groundtruth boxes if available
        6. Displays combined visualization
        7. Handles exit condition (Q key press)
        
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
                cv.imshow("Detection", image)
                frame_idx+=1
                if cv.waitKey(25) & 0xFF == ord('q'):
                    break
        except Exception as e:
            if self.writer:
                self.writer.clear()
            raise Exception(e)
        finally:
            cv.destroyAllWindows()

    def silent_show(self):
        """
        Processes frames sequentially with the following workflow:
        1. Retrieves next frame from data reader
        2. Runs object detection
        3. Write retrieved data from detection and write in file if available
        4. Progress bar show proccess of work detector
        5. Handles exit condition (Q key press)
        """
        try:
            self.__create_progress_bar()
            frame_idx = 0
            for image in self.datareader:
                if image is None:
                    break
                for box in self.detector.detect(image):
                    if self.writer:
                        self.writer.write((frame_idx, *box))
                frame_idx += 1
                self.__update_progress_bar()
                if 0xFF == ord('q'):
                    break
        except Exception as e:
            if self.writer:
                self.writer.clear()
            raise Exception(e)
        finally:
            self.__close_progress_bar()

    def __create_progress_bar(self):
        """Initialize progress bar with total frame count"""
        total_frames = self.datareader.get_total_frames()
        self.progress_bar = tqdm(
            total=total_frames,
            desc="Processing frames",
            unit="frame",
            dynamic_ncols=True
        )

    def __update_progress_bar(self):
        """Update progress bar state"""
        if self.progress_bar:
            self.progress_bar.update(1)

    def __close_progress_bar(self):
        """Properly close progress bar"""
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None

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

    def __get_groundtruth_bboxes(self, frame_idx):
        """
        Internal method: Filter groundtruth boxes for current frame.
        
        :param frame_idx: Current frame index
        :return:list: Groundtruth boxes for specified frame in format [label, x1, y1, x2, y2]
        """
        return [item[1:] for item in self.gt_layout if item[0] == frame_idx]
