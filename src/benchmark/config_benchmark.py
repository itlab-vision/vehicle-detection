"""
Benchmark Configuration Module

Defines experiment parameters and configures components required for the detection pipeline.
Includes setup of data readers, detector, visualizer, writer, and groundtruth reader.
"""
from pathlib import Path
from dataclasses import dataclass

from src.detector_pipeline.detector_pipeline import PipelineComponents
from src.gui_application.visualizer import BaseVisualizer
from src.utils.data_reader import CsvGTReader
from src.utils.frame_data_reader import FrameDataReader
from src.utils.writer import Writer
from src.vehicle_detector.detector import Detector


@dataclass
class ExperimentParameters:
    """
    Stores parameters for the detection experiment.

    :param input_data_path: Path to directory with images or a video file.
    :param groundtruth_path: Path to the .csv file with groundtruth annotations.
    :param output_path: Path to the directory where results will be saved
                        (e.g., result tables, plots).
    :param mode: Type of input data ('image' for image sets or 'video').
    """

    input_data_path: str
    groundtruth_path: str
    output_path: str
    mode: str = 'image'


def config_pipeline_components(
        config_params: dict,
        batch_size: int,
        output_path: Path,
        exp_params: ExperimentParameters):
    """
    Configures and initializes components for the detection pipeline.

    :param config_params: Dictionary containing model configuration parameters.
    :param batch_size: Number of frames to be processed in one batch.
    :param output_path: Path to temporary or final output (used by writer).
    :param exp_params: ExperimentParameters object containing data and mode info.
    :return: Initialized PipelineComponents object.
    """
    reader = FrameDataReader.create(exp_params.mode, exp_params.input_data_path,
                                    batch_size)
    param_detect = {
        'scale': config_params['scale'],
        'size': config_params['size'],
        'mean': config_params['mean'],
        'swapRB': config_params['swapRB']
    }

    param_adapter = {
        'confidence': config_params['confidence'],
        'nms_threshold': config_params['nms_threshold']
    }

    paths = {
        'path_weights': config_params['path_weights'],
        'path_config': config_params['path_config'],
        'path_anchors': config_params['path_anchors']
    }

    detector = Detector.create(config_params['adapter_name'], config_params['path_classes'],
                               paths, param_adapter, param_detect)

    visualizer = BaseVisualizer.create(silent=True)
    writer = Writer.create(output_path)
    gt_reader = CsvGTReader(exp_params.groundtruth_path)

    return PipelineComponents(
        reader, detector, visualizer, writer, gt_reader
    )
