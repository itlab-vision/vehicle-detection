"""
Some
"""
from dataclasses import dataclass
import numpy
from src.utils.frame_data_reader import FrameDataReader
from src.utils.writer import Writer
from src.vehicle_detector.detector import Detector
import src.utils.data_reader as dr
from src.gui_application.visualizer import BaseVisualizer

@dataclass
class PipelineComponents:
    """Container for all pipeline components"""
    reader: FrameDataReader
    detector: Detector
    visualizer: BaseVisualizer
    writer: Writer = None
    gt_reader: dr.DataReader = None


class DetectionPipeline:
    """Some"""
    def __init__(self, components: PipelineComponents):
        """Some"""
        if (components.visualizer or components.reader or components.detector) is None:
            raise ValueError("Missing pipeline components")

        self.components = components
        self.gtboxes = None

    def run(self):
        """Some"""
        try:
            with self.components.reader as reader:
                self.components.visualizer.initialize(reader.get_total_images())
                if self.components.gt_reader:
                    self.gtboxes = self._get_gtbboxes(self.components.gt_reader.read())

                for frame_idx, frame in enumerate(reader):
                    self._process_frame(frame_idx, frame)
                    self.components.visualizer.update_progress()
                    if self._should_exit():
                        break

        except Exception as e:
            self._handle_error(e)
        finally:
            self._finalize()

    def _process_frame(self, frame_idx: int, frame: numpy.ndarray):
        """Some"""
        detections = self.components.detector.detect(frame)

        if self.components.writer:
            self._write_results(frame_idx, detections)

        self.components.visualizer.visualize_frame(
            frame, detections,
            self.gtboxes[frame_idx]
        )

    def _write_results(self, frame_idx: int, detections: list[tuple]):
        """Some"""
        self.components.writer.write((frame_idx,) + det for det in detections)

    def _should_exit(self):
        """Some"""
        return self.components.visualizer.check_exit()

    def _handle_error(self, error: Exception):
        """Some"""
        if self.components.writer:
            self.components.writer.clear()
        raise RuntimeError(error)

    def _finalize(self):
        """Some"""
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
