"""

"""
import numpy
import cv2 as cv
from utils.frame_data_reader import FrameDataReader
from utils.writer import Writer
from vehicle_detector.detector import Detector
import utils.data_reader as dr
from gui_application.visualizer import Visualize
from accuracy_checker.accuracy_checker import AccuracyCalculator

class PipelineComponents:
    """"""
    def __init__(self, reader:FrameDataReader, detector:Detector, visualizer:Visualize, 
                writer: Writer=None , gt_reader:dr.DataReader=None, accuracy_calc:AccuracyCalculator=None):
        """"""
        self.reader = reader
        self.detector = detector
        self.visualizer = visualizer
        self.writer = writer
        self.gt_reader = gt_reader
        self.accuracy_calc = accuracy_calc

class DetectionPipeline:
    """"""
    def __init__(self, components:PipelineComponents):
        """"""
        self.components = components
    
    def run(self):
        """"""
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
        """"""
        detections = self.components.detector.detect(frame)

        if self.components.writer:
            self._write_results(frame_idx, detections)

        if self.components.visualizer:
            self._visualize(frame, detections)

    def _write_results(self, frame_idx: int, detections: list[tuple]):
        """"""
        self.components.writer.write((frame_idx,) + det for det in detections)

    def _visualize(self, frame_idx: int, frame: numpy.ndarray, detected: list[tuple]):
        """"""
        for box in detected:
            self.components.visualizer._draw_box(frame, box, (255, 0, 0))
        
        if self.components.gt_reader:
            gt_boxes = [item[1:] for item in self.components.gt_reader if item[0] == frame_idx]
            for box in gt_boxes:
                self.components.visualizer._draw_box(frame, box, (0, 255, 0))
        self._show_frame(frame)

    def _show_frame(self, frame: numpy.ndarray):
        """"""
        if self.components.visualizer:
            cv.imshow("Detection Output", frame)

    def _should_exit(self):
        """"""
        return cv.waitKey(25) & 0xFF == ord('q')

    def _handle_error(self, error: Exception):
        """"""
        if self.components.writer:
            self.components.writer.clear()
        raise RuntimeError(error)

    def _cleanup(self):
        """"""
        cv.destroyAllWindows()