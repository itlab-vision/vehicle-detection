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
import time
import sys
from abc import ABC, abstractmethod
from tqdm import tqdm
import cv2 as cv
import numpy


class BaseVisualizer(ABC):
    """Some"""
    @abstractmethod
    def initialize(self, total_frames: int):
        """Some"""

    @abstractmethod
    def update_progress(self):
        """Some"""

    @abstractmethod
    def visualize_frame(self, frame: numpy.ndarray,
                     detections: list, ground_truth: list = None):
        """Some"""

    @abstractmethod
    def check_exit(self):
        """Some"""

    @abstractmethod
    def finalize(self):
        """Some"""


class GUIVisualizer(BaseVisualizer):
    """
    Visualization controller for detection/groundtruth comparison.

    Handles frame iteration, bounding box drawing, and display management.
    Uses different colors for detected boxes (blue) and groundtruth boxes (green).
    """

    def __init__(self):
        """
        Initialize visualization components with data sources.
        """
        self.start_time = time.time()
        self.window_name = "Detection Output"
        self.progress_bar = None

    def initialize(self, total_frames: int):
        """Some"""
        self.progress_bar = tqdm(
            total=total_frames,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            colour='green',
            dynamic_ncols=True,
            unit='frame',
            unit_scale=True,
            position=0
        )

        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)

    def update_progress(self):
        """No needed for GUI"""
        self.progress_bar.update(1)

    def visualize_frame(self, frame: numpy.ndarray,
                     detections: list, ground_truth: list = None):
        """Some"""
        for box in detections:
            self._draw_box(frame, box, (255, 0, 0))

        for box in ground_truth:
            self._draw_box(frame, box, (0, 255, 0))

        cv.imshow(self.window_name, frame)

    @staticmethod
    def _draw_box(image: numpy.ndarray, box: tuple, color: tuple):
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
        cv.putText(image, label, (x1 + 10, y1 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.putText(image, confidence, (x1 - 10, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

    def check_exit(self):
        """Some"""
        return cv.waitKey(5) & 0xFF == ord('q')

    def finalize(self):
        """Release resourses"""
        cv.destroyAllWindows()
        self.progress_bar.close()
        self.progress_bar = None


class CLIVisualizer(BaseVisualizer):
    """
    Visualization controller for detections.
    Handles frame iteration and custom progress bar, print all detections.
    """
    def __init__(self):
        self.start_time = 0
        self.frame_idx = 0
        self.total_frames = 0

    def initialize(self, total_frames: int):
        """Some"""
        self.start_time = time.time()
        self.total_frames = total_frames
        print("Starting processing...")

    def update_progress(self):
        """Some"""

        elapsed = time.time() - self.start_time
        fps = self.frame_idx / elapsed if elapsed > 0 else 0
        status = (f"\rProcessed {self.frame_idx}/{self.total_frames} frames | "
            f"Elapsed: {elapsed:.1f}s | FPS: {fps:.1f} | "
            f"ETA: {(self.total_frames - self.frame_idx)/fps:.3f}s\n")
        sys.stdout.write("\r\033[K" + status)
        sys.stdout.flush()

    def visualize_frame(self, frame: numpy.ndarray,
                     detections: list, ground_truth: list = None):
        """Some"""
        if frame is None:
            return

        self._print_frame_header(self.frame_idx)
        self._print_bbox(detections, "Detections")

        self.frame_idx += 1

    def _print_frame_header(self, frame_idx: int):
        print(f"\n=== Frame {frame_idx} ===")

    def _print_bbox(self, boxes: list, title: str):
        if not boxes:
            return

        print(f"{title}:")
        for box in boxes:
            label = box[0]
            coords = list(map(int, box[1:5]))
            conf = f", Confidence: {box[5]:.2f}" if len(box) > 5 else ""
            print(f"  {label}: ({coords[0]}, {coords[1]})"
                 f" - ({coords[2]}, {coords[3]}){conf}")

    def check_exit(self):
        return 0xFF == ord('q')

    def finalize(self):
        elapsed = time.time() - self.start_time
        print(f"\n\nProcessing completed in {elapsed:.2f} seconds")
