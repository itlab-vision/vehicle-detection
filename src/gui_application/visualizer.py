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
from ..utils.frame_data_reader import FrameDataReader
from ..vehicle_detector.detector import FakeDetector
class Visualize:
    """Visualization controller for detection/groundtruth comparison.

    Handles frame iteration, bounding box drawing, and display management.
    Uses different colors for detected boxes (blue) and groundtruth boxes (green).

    Args:
        datareader (FrameDataReader): Input source for frames (video/images)
        detector (FakeDetector): Object detection implementation
        gt_data (list): Groundtruth data in format [[frame_idx, label, x1, y1, x2, y2], ...]

    Attributes:
        datareader (FrameDataReader): Frame input source
        detector (FakeDetector): Detection component
        gt_layout (list): Loaded groundtruth annotations
    """
    def __init__(self, datareader:FrameDataReader, detector:FakeDetector, gt_data:list):
        """Initialize visualization components with data sources."""
        self.datareader = datareader
        self.detector = detector
        self.gt_layout = gt_data

    def show(self):
        """Main visualization loop.
        
        Processes frames sequentially with the following workflow:
        
        1. Retrieves next frame from data reader
        2. Runs object detection and draws blue bounding boxes
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
        """Internal method: Draw single bounding box with label.
        
        Args:
            image: Input frame matrix
            box: Detection tuple (label, x1, y1, x2, y2)
            color: BGR tuple for box/label color
        """
        label, x1, y1, x2, y2 = box
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv.putText(image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.imshow("Image", image)

    def __get_groundtruth_bboxes(self, frame_idx):
        """Internal method: Filter groundtruth boxes for current frame.
        
        Args:
            frame_idx: Current frame index
            
        Returns:
            list: Groundtruth boxes for specified frame in format [label, x1, y1, x2, y2]
        """
        return [item[1:] for item in self.gt_layout if item[0] == frame_idx]
