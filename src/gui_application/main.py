from datareader import DataReader
from visualizer import Visualize
import datetime
import argparse
import os
from PIL import Image

import cv2 as cv
import numpy as np

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--mode',
                        help='Mode (\'image\', \'video\', \'img-pack\')',
                        type=str,
                        dest='mode',
                        default='image')
    parser.add_argument('-v', '--video',
                        help='Path to a video file',
                        type=str,
                        dest='video_path',
                        default='samplevideo.mp4')
    parser.add_argument('-i', '--image',
                        help='Path to an images',
                        type=str,
                        dest='images_path',
                        default="../DLMini/data/imgs_MOV03478/")
    parser.add_argument('-l', '--layout',
                        help='Path to a dir of layout',
                        type=str,
                        dest='dir_layout_path',
                        default="../DLMini/layout/mov03478.txt")
    parser.add_argument('-m', '--model',
                        help='Path to a model',
                        type=str,
                        dest='model_path',
                        default=None)

    
    args = parser.parse_args()
    return args

#некоторое подобие main 
def main():
    args = cli_argument_parser()
    reader = DataReader.create(args)
    adapter = None
    detector = None
    
    visual = Visualize(reader, detector)
    visual.show()
    
if __name__ == "__main__":
    main()