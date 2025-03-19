"""
Some
"""
from dataclasses import dataclass
import numpy
import cv2 as cv
from src.utils.frame_data_reader import FrameDataReader
from src.utils.writer import Writer
from src.vehicle_detector.detector import Detector
import src.utils.data_reader as dr
from src.gui_application.visualizer import Visualizer
from src.gui_application.cli_visualizer import CLIVisualize

@dataclass
class PipelineComponents:
    """Container for all pipeline components"""
    reader: FrameDataReader
    detector: Detector
    visualizer: Visualizer
    cli_visual: CLIVisualize
    writer: Writer = None
    gt_reader: dr.DataReader = None


class DetectionPipeline:
    """Some"""
    def __init__(self, components:PipelineComponents):
        """Some"""
        self.components = components
        self.progress_bar = None

    def run(self):
        """Some"""
        try:
            with self.components.reader as reader:
                for frame_idx, frame in enumerate(reader):
                    self._process_frame(frame_idx, frame)
                    if self._should_exit():
                        break

        except Exception as e:
            self._handle_error(e)
        finally:
            self._cleanup()

    def _process_frame(self, frame_idx: int, frame: numpy.ndarray):
        """Some"""
        detections = self.components.detector.detect(frame)

        if self.components.writer:
            self._write_results(frame_idx, detections)

        if self.components.visualizer:
            self._visualize(frame_idx, frame, detections)

        if self.components.cli_visual:
            self.components.cli_visual.update_status(frame_idx)
            self.components.cli_visual.print_detections(detections)

    def _write_results(self, frame_idx: int, detections: list[tuple]):
        """Some"""
        self.components.writer.write((frame_idx,) + det for det in detections)

    def _visualize(self, frame_idx: int, frame: numpy.ndarray, detections: list[tuple]):
        """Some"""
        for box in detections:
            self.components.visualizer.draw_box(frame, box, (255, 0, 0))

        if self.components.gt_reader:
            gt_boxes = self._parse_gtbbox(frame_idx)
            for box in gt_boxes:
                self.components.visualizer.draw_box(frame, box, (0, 255, 0))

        self.components.visualizer.show_frame(frame)

    def _parse_gtbbox(self, frame_idx: int):
        return [item[1:] for item in self.components.gt_reader.read() if item[0] == frame_idx]

    def _should_exit(self):
        """Some"""
        if self.components.visualizer:
            return cv.waitKey(5) & 0xFF == ord('q')
        return 0xFF == ord('q')

    def _handle_error(self, error: Exception):
        """Some"""
        if self.components.writer:
            self.components.writer.clear()
        raise RuntimeError(error)

    def _cleanup(self):
        """Some"""
        if self.components.visualizer:
            self.components.visualizer.cleanup()
