"""
Silent mode CLI application "Vehicle detector"
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.gui_application.visualizer import Visualize
from src.utils import data_reader as dr
from src.utils.frame_data_reader import FrameDataReader
from src.utils.writer import Writer
from src.vehicle_detector.detector import Detector
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
                        dest='groundtruth_path',
                        required=False)
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
                        required=False,
                        default=None)
    args = parser.parse_args()
    return args

def main():
    """
    Main execution function in silent mode.

    Initializes data reader, detector, and visualizer components based on CLI arguments.
    Shows visualization using the following workflow:
    
        1. Creates FrameDataReader based on input mode (video/image)
        2. Initializes a writer with 'write_path' implementation
        2. Initializes a detector with 'fake' implementation
        3. Loads groundtruth data if provided
        4. Configures visualizer with reader, detector, and groundtruth
        5. 

    Requires:
        - Either video_path or images_path argument must match the specified mode
        - model_path must point to a valid model file
    """
    try:
        args = cli_argument_parser()
        reader = FrameDataReader.create(args.mode, (args.video_path or args.images_path))
        writer = Writer.create(args.write_path)
        detector = Detector.create( "fake" )
        gtreader = dr.FakeGTReader(args.groundtruth_path)
        accurcheck = AccuracyCalculator()
        visualizer = Visualize(reader, writer, detector, gtreader.read())
        visualizer.silent_show()
        accurcheck.load_detections(args.write_path)
        accurcheck.load_groundtruths(args.groundtruth_path)
        print(f"TPR: {accurcheck.calc_tpr()}")
        print(f"FDR: {accurcheck.calc_fdr()}")
        print(f"MAP: {accurcheck.calc_map()}")

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
