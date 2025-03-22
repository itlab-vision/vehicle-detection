"""
Computer Vision Visualization Module

Provides abstract and concrete implementations for visualizing object detection results.
Supports both GUI-based (OpenCV) and CLI-based visualization with progress tracking.
"""
import time
import sys
from abc import ABC, abstractmethod
from tqdm import tqdm
import cv2 as cv
import numpy


class BaseVisualizer(ABC):
    """Abstract base class defining visualization interface."""

    @abstractmethod
    def initialize(self, total_frames: int):
        """
        Initialize visualization resources.

        :param total_frames: Total number of frames to process
        """

    @abstractmethod
    def update_progress(self):
        """Update progress tracking display."""

    @abstractmethod
    def visualize_frame(self, frame: numpy.ndarray,
                     detections: list, ground_truth: list = None):
        """
        Render frame with detection annotations.

        :param frame: Input image array in BGR format
        :param detections: List of detected objects in format:
                        (label, x1, y1, x2, y2[, confidence])
        :param ground_truth: List of ground truth boxes in format:
                         (label, x1, y1, x2, y2)
        """

    @abstractmethod
    def check_exit(self):
        """
        Check for early termination request.

        :return bool: True if termination requested, False otherwise
        """

    @abstractmethod
    def finalize(self):
        """Release visualization resources."""

    @staticmethod
    def create(silent: bool):
        """
        Factory method for visualizer instances.

        :param silent: If True, creates CLI visualizer; otherwise GUI
        """
        if silent:
            return CLIVisualizer()
        return GUIVisualizer()


class GUIVisualizer(BaseVisualizer):
    """
    GUI visualization using OpenCV with real-time annotations and progress bar.

    Features:
    - Bounding box rendering (blue for detections, green for ground truth)
    - Confidence score display
    - Interactive window with keyboard controls
    - Frame rate statistics
    """

    def __init__(self):
        """
        Initialize visualization components with data sources.
        """
        self.start_time = time.time()
        self.window_name = "Detection Output"
        self.progress_bar = None

    def initialize(self, total_frames: int):
        """
        Initialize OpenCV window and progress bar.
        """
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
        """Advance progress bar by one frame."""
        self.progress_bar.update(1)

    def visualize_frame(self, frame: numpy.ndarray,
                     detections: list, ground_truth: list = None):
        """Render frame with bounding boxes and text annotations."""
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
        """Check for Q key press to terminate visualization."""
        return cv.waitKey(25) & 0xFF == ord('q')

    def finalize(self):
        """Cleanup OpenCV resources and progress bar."""
        cv.destroyAllWindows()
        self.progress_bar.close()
        self.progress_bar = None


class CLIVisualizer(BaseVisualizer):
    """Command-line visualization with textual output and statistics.
    
    Features:
    - Frame-by-frame detection reports
    - Processing statistics (FPS, ETA)
    - Minimal resource usage
    """

    def __init__(self):
        self.start_time = 0
        self.frame_idx = 0
        self.total_frames = 0

    def initialize(self, total_frames: int):
        """Initialize processing timer."""
        self.start_time = time.time()
        self.total_frames = total_frames
        print("Starting processing...")

    def update_progress(self):
        """Update console progress display."""

        elapsed = time.time() - self.start_time
        fps = self.frame_idx / elapsed if elapsed > 0 else 0
        status = (f"\rProcessed {self.frame_idx}/{self.total_frames} frames | "
            f"Elapsed: {elapsed:.1f}s | FPS: {fps:.1f} | "
            f"ETA: {(self.total_frames - self.frame_idx)/fps:.3f}s\n")
        sys.stdout.write("\r\033[K" + status)
        sys.stdout.flush()

    def visualize_frame(self, frame: numpy.ndarray,
                     detections: list, ground_truth: list = None):
        """Print frame detection details to console."""

        self._print_frame_header(self.frame_idx)
        self._print_bbox(detections, "Detections")

        self.frame_idx += 1

    def _print_frame_header(self, frame_idx: int):
        """Internal: Print frame separator."""
        print(f"\n=== Frame {frame_idx} ===")

    def _print_bbox(self, boxes: list, title: str):
        """Internal: Print detection details."""
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
        """CLI visualizer doesn't support interactive termination."""
        return False

    def finalize(self):
        """Print final statistics."""
        elapsed = time.time() - self.start_time
        print(f"\n\nProcessing completed in {elapsed:.2f} seconds")
