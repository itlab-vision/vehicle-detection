"""
CLI Application "Vehicle Detector"

A command-line interface application for detecting vehicles in images or video streams. 
Supports multiple visualization modes, result recording, and accuracy calculations against 
groundtruth data.

Key Features:
- Vehicle detection in image/video inputs
- GUI/CLI visualization options
- Result export capabilities
- Precision metrics calculation (TPR, FDR, mAP)
- Flexible pipeline configuration

Modules:
- data_reader: Handles input data loading
- detector: Contains vehicle detection logic
- visualizer: Provides visualization interfaces
- writer: Manages result storage
- accuracy_checker: Computes detection accuracy metrics
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.gui_application.visualizer import BaseVisualizer
from src.utils import data_reader as dr
from src.utils.frame_data_reader import FrameDataReader
from src.utils.writer import Writer
from src.vehicle_detector.detector import Detector
from src.detector_pipeline.detector_pipeline import PipelineComponents, DetectionPipeline
from src.accuracy_checker.accuracy_checker import AccuracyCalculator


def cli_argument_parser():
    """
    Parse command-line arguments for the visualizer application.

    Defines and parses arguments for specifying input mode (image/video), file paths,
    groundtruth data, and model path. Ensures mutual exclusivity between video and image paths
    based on the selected mode.

    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            - mode (str): Input mode ('image' or 'video'), required
            - video_path (str): Path to video file (required if mode is 'video')
            - images_path (str): Path to image directory (required if mode is 'image')
            - groundtruth_path (str): Path to groundtruth data file, optional
            - model_path (str): Path to model file, required

    Raises:
        argparse.ArgumentError:
            If required arguments are missing or invalid combinations are provided
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--mode',
                        help='Mode (\'image\', \'video\')',
                        type=str,
                        dest='mode',
                        choices=['image', 'video'],
                        required=True)
    parser.add_argument('-v', '--video',
                        help='Path to a video file',
                        type=str,
                        dest='video_path')
    parser.add_argument('-i', '--image',
                        help='Path to images',
                        type=str,
                        dest='images_path')
    parser.add_argument('-g', '--groundtruth',
                        help='Path to a file of groundtruth',
                        type=str,
                        dest='groundtruth_path')
    parser.add_argument('-m', '--model',
                        help='Path to a model',
                        type=str,
                        dest='model_path',
                        required=True,
                        default=None)
    parser.add_argument('-w', '--write',
                        help='Full path to the file to write.',
                        type=str,
                        dest='write_path',
                        default=None)
    parser.add_argument('-s', '--silent',
                        help='Set silent mode of program',
                        action='store_true',
                        dest='silent_mode')
    parser.add_argument('-b', '--batches',
                        help='Set size of image batch',
                        type=int,
                        dest='batch_size',
                        default=None)

    args = parser.parse_args()
    return args


def config_main(args: argparse.Namespace):
    """
    Configure pipeline components.

    :param argparse.Namespace: Parsed command-line arguments

    :return PipelineComponents: Configured pipeline objects with GUI visualizer
    """
    return PipelineComponents(
            reader = FrameDataReader.create(args.mode, (args.video_path or args.images_path), args.batch_size),
            detector = Detector.create( "fake" ),
            visualizer = BaseVisualizer.create(args.silent_mode),
            writer = Writer.create(args.write_path) if args.write_path else None,
            gt_reader = dr.CsvGTReader(args.groundtruth_path) if args.groundtruth_path else None)


def main():
    """"
    Main execution flow for vehicle detection application.

    Workflow:
    1. Parse command-line arguments
    2. Configure detection pipeline
    3. Run vehicle detection process
    4. Calculate and display accuracy metrics (if groundtruth provided)
    5. Handle and report runtime errors

    Requires:
    - Valid input path matching selected mode
    - Accessible detection model file
    - Compatible groundtruth format (when provided)
    """
    try:
        args = cli_argument_parser()

        components = config_main(args)

        pipeline = DetectionPipeline(components)
        pipeline.run()

        if all([args.groundtruth_path, args.write_path]):
            accur_calc = AccuracyCalculator()
            accur_calc.load_detections(args.write_path)
            accur_calc.load_groundtruths(args.groundtruth_path)
            print (f"TPR: {accur_calc.calc_tpr()}\n"
                f"FDR: {accur_calc.calc_fdr()}\n"
                f"MAP: {accur_calc.calc_map()}")

    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
