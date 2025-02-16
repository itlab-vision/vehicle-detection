from datareader import DataReader
import visualizer as visual
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
    reader = DataReader.create(args)
    adapter = None
    detector = visual.PseudoDetector(args.groundtruth_path) # doesn't get groundtruth_path for real, this for only demonstration. Fix later.
    visualizer = visual.Visualize(reader, detector)
    visualizer.show()
    
if __name__ == '__main__':
    main()