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
class BatchesTimings:
    """Data structure for storing batch processing timing metrics.

    Attributes:
        preprocess_time (List[float]):
            List of preprocessing durations (in seconds) for each batch
        inference_time (List[float]):
            List of model inference durations (in seconds) for each batch
        postprocess_time (List[float]):
            List of postprocessing durations (in seconds) for each batch
    """
    preprocess_time:    list[float]
    inference_time:     list[float]
    postprocess_time:   list[float]


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
        self.batches_timings = BatchesTimings([], [], [])
        self.current_batch_start_idx = 0
        self.batch_size = 1
        self.total_batches = 0

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

    def get_batches_timings(self):
        """
        :return BatchTimimg: tuple of lists of all processed batches
        """
        return self.batches_timings

    def _initialize_processing(self, reader: FrameDataReader):
        """Prepare processing components and load ground truth if available."""

        total_frames = reader.get_total_images()
        batch_size = getattr(reader, 'batch_size', 1)
        self.total_batches = (total_frames + batch_size - 1) // batch_size

        self.components.visualizer.initialize(self.total_batches)
        self.batch_size = batch_size

        if self.components.gt_reader:
            raw_gt = self.components.gt_reader.read()
            self.gtboxes = self._organize_ground_truth(raw_gt)

    def _process_batch(self, batch_idx: int, batch: list[np.ndarray]):
        """
        Process a single batch through the detection pipeline.

        :param batch_idx: Index of the current batch
        :param batch: Image batch data in list of numpy array format
        """
        batch_detects, preproc_time, inference_time, postproc_time \
            = self.components.detector.detect(batch)

        if batch_idx < self.total_batches - 1:
            self.batches_timings.preprocess_time.append(preproc_time)
            self.batches_timings.inference_time.append(inference_time)
            self.batches_timings.postprocess_time.append(postproc_time)

        self.components.visualizer.batch_metrics(
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
        self.components.writer.write(list((frame_idx, *det) for det in detections))

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
