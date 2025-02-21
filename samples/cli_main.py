import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.gui_application.visualizer import Visualize
from src.utils.data_reader import GroundtruthReader, FakeGroundtruthReader
from src.utils.frame_data_reader import FrameDataReader
from src.vehicle_detector.detector import Detector
import datetime
import argparse
import os
from PIL import Image

import cv2 as cv
import numpy as np

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

#некоторое подобие main 
def main():
    args = cli_argument_parser()
    reader = FrameDataReader.create( args.mode, (args.video_path or args.images_path) )
    adapter = None
    detector = Detector.create( "fake" )
    visualizer = Visualize( reader, detector, FakeGroundtruthReader().read(args.groundtruth_path) )
    visualizer.show()
    
if __name__ == '__main__':
    main()