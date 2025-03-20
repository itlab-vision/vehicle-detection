"""
Computer Vision CLI Visualization Module

Provides functionality for visualizing object detection results alongside ground truth data.
Handles both image sequences and video inputs with the following components:
- Frame data reading (images/video)
- Object detection implementation

Dependencies:
- OpenCV (cv2) for image processing and display
- time and sys for output progressing in console
- FrameDataReader for getting images
- Writer for write detections in file
- Detector for getting detections in image
"""
import time
import sys
from src.utils.frame_data_reader import FrameDataReader
from src.utils.writer import Writer
from src.vehicle_detector.detector import Detector

class CLIVisualize:
    """
    Visualization controller for detections.

    Handles frame iteration and custom progress bar, print all detections.
    """

    def __init__(self, datareader:FrameDataReader, writer:Writer, detector:Detector):
        """
        Initialize visualization components with data sources.

        :param datareader: Input source for frames (video/images)
        :param detector: Detection component
        """
        self.datareader = datareader
        self.writer = writer
        self.detector = detector
        self.start_time = None

        self.last_update = 0

    def show(self):
        """
        Processes frames sequentially with the following workflow:
        1. Retrieves next frame from data reader
        2. Runs object detection
        3. Write retrieved data from detection and write in file if available
        4. Progress bar show proccess of work detector
        5. Handles exit condition (Q key press)
        """
        try:
            self.start_time = time.time()
            frame_idx = 0

            for frame in self.datareader:
                current_time = time.time()
                if frame is None:
                    break
                detections = self.detector.detect(frame)

                self._print_detections(detections)

                if current_time - self.last_update >= 0.5:
                    self._update_status(frame_idx)
                    self.last_update = current_time

                for box in detections:
                    if self.writer:
                        self.writer.write((frame_idx, *box))

                frame_idx += 1

                if 0xFF == ord('q'):
                    break
        except Exception as e:
            if self.writer:
                self.writer.clear()
            raise Exception(e)
        finally:
            self._print_final_status(frame_idx)

    def _update_status(self, frame_idx: int):
        """
        Update string status

        :param frame_idx: current id of frame
        """
        elapsed = time.time() - self.start_time
        fps = frame_idx / elapsed if elapsed > 0 else 0
        status = (
            f"Processing frame: {frame_idx} "
            f"[{elapsed:.1f}s, {fps:.1f}fps]"
        )
        sys.stdout.write("\r\033[K" + status)
        sys.stdout.flush()

    def _print_detections(self, detections: list[tuple]):
        """
        print info about detections
        
        :param detections: list of tuples that got by detector
        """
        for box in detections:
            label = box[0]
            coords = tuple(map(int, box[1:5]))
            detection_str = (
                f"\n  {label} {coords[0]},{coords[1]}-{coords[2]},{coords[3]}"
            )
            sys.stdout.write(detection_str)
            sys.stdout.flush()

    def _print_final_status(self, frame_idx: int):
        """
        Print final result

        :param frame_idx: current id of frame
        """
        elapsed = time.time() - self.start_time
        fps = frame_idx / elapsed if elapsed > 0 else 0
        print(
            f"\nProcessing completed: {frame_idx} frames "
            f"in {elapsed:.1f}s ({fps:.1f} fps)"
        )
