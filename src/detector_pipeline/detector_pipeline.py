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
        self.progress_bar = None

    def run(self):
        """Some"""
        try:
            with self.components.reader as reader:
                self.components.visualizer.initialize(reader.get_total_images())
                if (self.components.gt_reader):
                    self.gtboxes = self.components.gt_reader.read()

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
            self._get_gtbbox(frame_idx)
        )

    def _write_results(self, frame_idx: int, detections: list[tuple]):
        """Some"""
        self.components.writer.write((frame_idx,) + det for det in detections)

    def _get_gtbbox(self, frame_idx: int):
        """Some"""
        if not self.components.gt_reader:
            return []
        return [item[1:] for item in self.gtboxes if item[0] == frame_idx]

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
