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
    def initialize(self, total_batches: int):
        """
        Initialize visualization resources.

        :param total_batches: Total number of frames to process
        """

    @abstractmethod
    def batch_metrics(self, batch_idx: int, preproc_time: float,
                      inference_time: float, postproc_time: float):
        """
        Handle batch processing start.

        :param batch_idx: Current batch index
        :param preproc_time: Batch preprocessing time in seconds
        :param inference_time: Model inference time in seconds
        :param postproc_time: Postprocessing time in seconds
        """

    @abstractmethod
    def batch_end(self):
        """Handle batch processing completion."""

    @abstractmethod
    def visualize_frame(self, frame_idx: int, frame: numpy.ndarray,
                        detections: list[tuple], ground_truth: list[tuple]):
        """
        RVisualize detection results on a single frame.

        :param frame_idx: Global frame index
        :param frame: Input image array in BGR format
        :param detections: List of detected objects in format:
                        (label, x1, y1, x2, y2, confidence)
        :param ground_truth: List of ground truth boxes in format:
                         (label, x1, y1, x2, y2)
        """

    @abstractmethod
    def check_exit(self):
        """
        Check for early termination request.

        :return bool: True if processing should terminate, False otherwise
        """

    @abstractmethod
    def finalize(self):
        """Release visualization resources."""

    @staticmethod
    def create(silent: bool):
        """
        Factory method for visualizer instances.

        :param silent: If True, creates CLI visualizer; otherwise GUI

        :return BaseVisualizer: Configured visualizer instance
        """
        if silent:
            return CLIVisualizer()
        return GUIVisualizer()


class GUIVisualizer(BaseVisualizer):
    """GUI visualization using OpenCV with real-time display.

    Features:
    - Bounding box visualization (blue for detections, green for ground truth)
    - Confidence score display
    - Processing time statistics
    - Interactive controls
    - Progress tracking
    """

    BOX_COLORS = {
        'detection': (255, 0, 0),
        'ground_truth': (0, 255, 0)
    }

    def __init__(self):
        """
        Initialize visualization components with data sources.
        """
        self.fp = 4
        self.start_time = time.time()
        self.window_name = "Vehicle Detection Output"
        self.current_batch = 0
        self.progress_bar = None
        self.processing_times = {
            'preproc': 0.0,
            'inference': 0.0,
            'postproc': 0.0
        }

    def initialize(self, total_batches: int):
        """
        Initialize OpenCV window and progress bar.
        """
        self.progress_bar = tqdm(
            total=total_batches,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            colour='green',
            dynamic_ncols=True,
            unit_scale=True,
            unit='batch',
            position=0
        )


        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)

    def batch_metrics(self, batch_idx: int, preproc_time: float,
                      inference_time: float, postproc_time: float):
        """Record batch processing metrics."""
        self.current_batch = batch_idx
        self.processing_times.update({
            'preproc': preproc_time,
            'inference': inference_time,
            'postproc': postproc_time
        })

    def batch_end(self):
        """Update progress display."""
        self.progress_bar.update(1)

    def visualize_frame(self, frame_idx: int, frame: numpy.ndarray,
                        detections: list[tuple], ground_truth: list[tuple]):
        """Render frame with bounding boxes and text annotations."""
        self._draw_processing_info(frame_idx, frame)

        for det in detections:
            self._draw_bbox(frame, det, 'detection')

        for gt in ground_truth:
            self._draw_bbox(frame, gt, 'ground_truth')

        cv.imshow(self.window_name, frame)
        cv.waitKey(33)

    def _draw_bbox(self, frame: numpy.ndarray, box_data: tuple, box_type: str):
        """
        Internal method: Draw single bounding box with label.

        :param frame: Input frame matrix
        :param box: Detection tuple (label, x1, y1, x2, y2, confidence(detector) )
        :param box_type: BGR tuple for box/label color
        """
        label, x1, y1, x2, y2, *confidence = box_data
        color = self.BOX_COLORS[box_type]

        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if confidence:
            label += f" {confidence[0]:.2f}"

        cv.putText(frame, label, (x1, y1 - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_processing_info(self, frame_idx: int, frame: numpy.ndarray):
        """Draw processing metadata on frame."""
        info_text = (
            f"Batch: {self.current_batch} | Frame: {frame_idx} | "
            f"Pre: {self.processing_times['preproc']:.{self.fp}f}s | "
            f"Inference: {self.processing_times['inference']:.{self.fp}f}s | "
            f"Post: {self.processing_times['postproc']:.{self.fp}f}s"
        )
        cv.putText(frame, info_text, (10, 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def check_exit(self):
        """Check for Q key press to terminate visualization."""
        return cv.waitKey(1) & 0xFF == ord('q')

    def finalize(self):
        """Cleanup OpenCV resources and progress bar."""
        cv.destroyAllWindows()
        if self.progress_bar:
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
        self.total_batches = 0
        self.current_batch = 0
        self.processing_times = {
            'preproc': 0.0,
            'inference': 0.0,
            'postproc': 0.0
        }

    def initialize(self, total_batches: int):
        """Initialize processing timer."""
        self.start_time = time.time()
        self.total_batches = total_batches
        print(f"Starting processing of {total_batches} batches")

    def batch_metrics(self, batch_idx: int, preproc_time: float,
                    inference_time: float, postproc_time: float):
        """Record batch metrics."""
        self.current_batch = batch_idx
        self.processing_times.update({
            'preproc': preproc_time,
            'inference': inference_time,
            'postproc': postproc_time
        })

        self._print_batch_header(batch_idx)

    def batch_end(self):
        """Update console progress display."""

        elapsed = time.time() - self.start_time
        sys.stdout.write(
            f"\rProcessed {self.current_batch + 1}/{self.total_batches} batches | "
            f"Elapsed: {elapsed:.1f}s | "
            f"Times (P/I/Po): {self.processing_times['preproc']:.4f}/"
            f"{self.processing_times['inference']:.4f}/"
            f"{self.processing_times['postproc']:.4f}s"
        )
        sys.stdout.flush()

    def visualize_frame(self, frame_idx: int, frame: numpy.ndarray,
                        detections: list, ground_truth: list = None):
        """Print frame detection details to console."""

        print(f"\tFrame {frame_idx}:")
        for det in detections:
            label, x1, y1, x2, y2, conf = det
            print(f"\t{label}: ({x1},{y1})-({x2},{y2}) @ {conf:.2f}")
        print()

    def _print_batch_header(self, batch_idx: int):
        """Internal: Print batch separator."""
        print(f"\nBatch {batch_idx}:")

    def check_exit(self):
        """CLI visualizer doesn't support interactive termination."""
        return False

    def finalize(self):
        """Print final statistics."""
        elapsed = time.time() - self.start_time
        print(f"\n\nProcessing completed in {elapsed:.2f} seconds")
