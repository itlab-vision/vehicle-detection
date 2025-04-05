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
        self.gtboxes = {}
        self.batch_size = 1
        self.current_batch_start_idx = 0

    def run(self):
        """
        Execute the main processing loop.
        
        Workflow:
        1. Initialize data sources and visualization
        2. Process frames in batches
        3. Handle detection and visualization
        4. Manage resource cleanup
        """
        try:
            with self.components.reader as reader:
                self._initialize_processing(reader)

                for batch_idx, batch in enumerate(reader):
                    self._process_batch(batch_idx, batch)
                    self.current_batch_start_idx += len(batch)

                    if self._should_exit():
                        break

        except Exception as e:
            self._handle_error(e)
        finally:
            self._finalize()

    def _initialize_processing(self, reader: FrameDataReader):
        """Prepare processing components and load ground truth if available."""
        self.components.visualizer.initialize(reader.get_total_batches())
        self.batch_size = getattr(reader, 'batch_size', 1)

        if self.components.gt_reader:
            raw_gt = self.components.gt_reader.read()
            self.gt_boxes = self._organize_ground_truth(raw_gt)

    def _process_batch(self, batch_idx: int, batch: list[np.ndarray]):
        """
        Process a single batch through the detection pipeline.

        :param batch_idx: Index of the current batch
        :param batch: Image batch data in list of numpy array format
        """
        detect_results = self.components.detector.detect(batch)
        batch_detects, preproc_time, inference_time, postproc_time = detect_results

        self.components.visualizer.batch_start(
            batch_idx, preproc_time,
            inference_time, postproc_time
        )

        for frame_offset, frame_detects in enumerate(batch_detects):
            frame_idx = batch_idx * self.batch_size + frame_offset

            if self.components.writer:
                self._write_results(frame_idx, frame_detects)

            self.components.visualizer.visualize_frame(
                frame_idx, batch[frame_offset],
                frame_detects, self.gtboxes.get(frame_idx, [])
            )

        self.components.visualizer.batch_end()

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
        raise RuntimeError(f"Pipeline processing failed: {error}")

    def _finalize(self):
        """Perform final cleanup operations."""
        self.components.visualizer.finalize()

    @staticmethod
    def _organize_ground_truth(gt_data: list[list]):
        """
        Internal method: Transform groundtruth data into frame-indexed dictionary.

        :param gt_data: List of groundtruth entries in format 
                    [frame_index, label, x1, y1, x2, y2]

        :return dict: Dictionary mapping frame indices to their groundtruth boxes
        """
        organized = {}
        for entry in gt_data:
            frame_idx = entry[0]
            organized.setdefault(frame_idx, []).append(entry[1:])
        return organized