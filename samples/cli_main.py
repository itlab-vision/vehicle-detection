import sys
import os
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.gui_application.visualizer import Visualize
from src.utils.data_reader import GroundtruthReader
from src.utils.frame_data_reader import FrameDataReader
from src.vehicle_detector.detector import Detector

def cli_argument_parser():
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
    args = parser.parse_args()
    return args

def main():
    args = cli_argument_parser()
    reader = FrameDataReader.create( args.mode, (args.video_path or args.images_path) )
    detector = Detector.create( "fake" )
    visualizer = Visualize( reader, detector, GroundtruthReader().read(args.groundtruth_path) )
    visualizer.show()

if __name__ == '__main__':
    main()
    