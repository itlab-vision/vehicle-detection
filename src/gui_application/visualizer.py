"""
Computer Vision Visualization Module

Provides functionality for visualizing object detection results alongside ground truth data.
Handles both image sequences and video inputs with the following components:
- Frame data reading (images/video)
- Object detection implementation
- Visualization of detection vs groundtruth comparisons

Dependencies:
- OpenCV (cv2) for image processing and display
- tqdm for progress bar processing
- FrameDataReader for getting images
- Writer for write detections in file
- Detector for getting detections in image
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
        self.gt_layout = self.__get_groundtruth_bboxes(gt_data)
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
            self.__create_progress_bar()
            for image in self.datareader:
                if image is None:
                    break
                for box in self.detector.detect(image):
                    self.__draw_box(image, box, (255, 112, 166))
                    if self.writer:
                        self.writer.write((frame_idx, *box))
                if self.gt_layout:
                    gt_bboxes = self.gt_layout.get(frame_idx)
                    for box in gt_bboxes:
                        self.__draw_box(image, box, (0, 255, 0))
                cv.imshow("Detection", image)
                self.__update_progress_bar()
                frame_idx+=1
                if cv.waitKey(25) & 0xFF == ord('q'):
                    break
        except Exception as e:
            if self.writer:
                self.writer.clear()
            raise Exception(e)
        finally:
            self.__close_progress_bar()
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

    @staticmethod
    def __get_groundtruth_bboxes(gt_data: list):
        """
        Internal method: Transform groundtruth data into frame-indexed dictionary.
        :param gt_data: List of groundtruth entries in format 
                    [[frame_index, label, x1, y1, x2, y2], ...]

        :return dict: Dictionary mapping frame indices to their groundtruth boxes
        """
        frame_dict = {}
        for entry in gt_data:
            frame_idx = entry[0]
            bbox_data = entry[1:]
            if frame_idx not in frame_dict:
                frame_dict[frame_idx] = []
            frame_dict[frame_idx].append(bbox_data)
        return frame_dict
