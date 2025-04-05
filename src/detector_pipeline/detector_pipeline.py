"""
Vehicle Detection Pipeline Implementation

This module provides classes and methods for managing a vehicle detection pipeline, 
including data processing, visualization, result recording, and error handling.
"""
from dataclasses import dataclass
import numpy as np
from src.utils.frame_data_reader import FrameDataReader
from src.utils.writer import Writer
from src.vehicle_detector.detector import Detector
import src.utils.data_reader as dr
from src.gui_application.visualizer import BaseVisualizer

@dataclass
class PipelineComponents:
    """Container class for essential pipeline components.

    Attributes:
        reader: Input data handler for frames (images/video)
        detector: Vehicle detection algorithm implementation
        visualizer: Visualization interface controller
        writer: Results output handler
        gt_reader: Ground truth data loader
    """
    reader: FrameDataReader
    detector: Detector
    visualizer: BaseVisualizer
    writer: Writer = None
    gt_reader: dr.DataReader = None


class DetectionPipeline:
    """
    Orchestrates the complete vehicle detection workflow.

    Manages frame processing, visualization, result recording, and error handling.
    Supports optional ground truth validation and early termination checks.
    """

    def __init__(self, components: PipelineComponents):
        """
        Initialize the detection pipeline with required components.
        
        :param components: Configured pipeline components
            
        :raise ValueError: If any essential component (reader, detector, visualizer) is missing
        """
        if (components.visualizer or components.reader or components.detector) is None:
            raise ValueError("Missing pipeline components")

        self.components = components
        self.gtboxes = None
        self.batch_size = 1

    def run(self):
        """
        Execute the image processing loop.
        
        Workflow:
        1. Initialize visualization and ground truth data (if available)
        2. Process frames or batches sequentially
        3. Handle detection, visualization, and result recording
        4. Manage cleanup and error handling
        """
        try:
            with self.components.reader as reader:
                self.components.visualizer.initialize(reader.get_total_batches())
                if hasattr(reader, 'batch_size'):
                    self.batch_size = getattr(reader, 'batch_size')

                if self.components.gt_reader:
                    self.gtboxes = self._get_gtbboxes(self.components.gt_reader.read())

                for batch_idx, batch in enumerate(reader):
                    self._process_batch(batch_idx, batch)

                    self.current_batch_start_idx += len(batch)

                    if self._should_exit():
                        break

        except Exception as e:
            self._handle_error(e)
        finally:
            self._finalize()

    def _process_batch(self, batch_idx: int, batch: list[np.ndarray]):
        """
        Process a single batch through the detection pipeline.

        :param batch_idx: Index of the current batch
        :param batch: Image batch data in list of numpy array format
        """
        batch_detects = self.components.detector.detect(batch)

        for frame_offset, frame_detects in enumerate(batch_detects):
            frame_idx = batch_idx * self.batch_size + frame_offset

            if self.components.writer:
                self._write_results(frame_idx, frame_detects)

            self.components.visualizer.visualize_frame(
                batch[frame_offset], frame_detects,
                self.gtboxes[frame_idx]
            )

        self.components.visualizer.update_progress()

        

    def _write_results(self, frame_idx: int, detections: list[tuple]):
        """
        Record detection results using the configured writer.

        :param frame_idx: Index of the current frame
        :param detections: List of detected objects in format 
                                    (label, confidence, x1, y1, x2, y2)
        """
        self.components.writer.write((frame_idx,) + det for det in detections)

    def _should_exit(self):
        """
        Check if processing should terminate early.

        :return bool: True if exit signal received from visualizer, False otherwise
        """
        return self.components.visualizer.check_exit()

    def _handle_error(self, error: Exception):
        """
        Handle pipeline errors and clean up resources.

        :param error: Caught exception during processing
        """
        if self.components.writer:
            self.components.writer.clear()
        raise RuntimeError(error)

    def _finalize(self):
        """
        Perform final cleanup operations after processing completes.
        """
        self.components.visualizer.finalize()

    @staticmethod
    def _get_gtbboxes(gt_data: list):
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
