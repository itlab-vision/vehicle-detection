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
import yaml

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

    parser.add_argument('-y', '--yaml',
                        type=str,
                        help = 'Path to a yaml file',
                        dest='yaml_file',
                        required=False)
    parser.add_argument('-t', '--mode',
                        help='Mode (\'image\', \'video\')',
                        type=str,
                        dest='mode',
                        choices=['image', 'video'],
                        required=False)
    parser.add_argument('-v', '--video',
                        help='Path to a video file',
                        type=str,
                        dest='video_path')
    parser.add_argument('-i', '--image',
                        help='Path to images',
                        type=str,
                        dest='images_path',
                        required=False)
    parser.add_argument('-g', '--groundtruth',
                        help='Path to a file of groundtruth',
                        type=str,
                        dest='groundtruth_path',
                        required=False)
    parser.add_argument('-m', '--model_name',
                        help='Model name (\'MobileNet\', \'YOLOv4\', \'rcnn_resnet50\', \'rcnn_resnet_v2\', \'YOLOv3_tiny\')',
                        type=str,
                        dest='model_name',
                        choices=['MobileNet', 'YOLOv4', 'rcnn_resnet50', 'rcnn_resnet_v2', 'YOLOv3_tiny'],
                        required=False,
                        default='MobileNet')
    parser.add_argument('-cl', '--path_classes',
                        help='Path to a file of classes',
                        type=str,
                        dest='path_classes',
                        required=False)
    parser.add_argument('-wp', '--path_weights',
                        help='Path to a file of weights',
                        type=str,
                        dest='path_weights',
                        required=False)
    parser.add_argument('-cp', '--path_config',
                        help='Path to a file of config',
                        type=str,
                        dest='path_config',
                        required=False)
    parser.add_argument('-c', '--confidence', 
                        help='Confidence threshold',
                        type=float, 
                        dest='conf',
                        required=False)
    parser.add_argument('-n', '--nms_threshold',
                        help='Overlap threshold',
                        type=float, 
                        dest='nms',
                        required=False)
    parser.add_argument('-s', '--scale',
                        help='Scale factor',
                        type=float, 
                        dest='scale',
                        required=False)
    parser.add_argument('-sz', '--size',
                        help='Size input model',
                        nargs=2,
                        type = int,
                        dest='size',
                        required=False)
    parser.add_argument('-me', '--mean',
                        help='The average picture',
                        nargs=3,
                        type = float,
                        dest='mean',
                        required=False)
    parser.add_argument('-sw', '--swapRB',
                        help='swapRB',
                        type=bool, 
                        dest='swapRB',
                        required=False)
    
    args = parser.parse_args()
    return args

def main():
    
    try:
        args = cli_argument_parser()
    
        if args.yaml_file != None:
        
            with open(args.yaml_file) as fh:
                data = yaml.safe_load(fh)
            data = data[0]
        
            reader = 0
            if data['mode'] == 'image':
                reader = FrameDataReader.create(data['mode'], data['images_path'])
            elif data['mode'] == 'video':
                reader = FrameDataReader.create(data['mode'], data['video_path'])
            
            adapter = None
            detector = Detector.create( data['model_name'], data['path_classes'], data['path_weights'], data['path_config'], float(data['confidence']),
                                       float(data['nms_threshold']), float(data['scale']), list(map(int, data['size'].split(' '))), 
                                       list(map(float, data['mean'].split(' '))), bool(data['swapRB']))
            visualizer = Visualize( reader, detector, GroundtruthReader().read(data['groundtruth_path']) )
            visualizer.show()
        
        else:
            reader = FrameDataReader.create( args.mode, (args.video_path or args.images_path) )
            adapter = None
            detector = Detector.create( args.model_name, args.path_classes, args.path_weights, args.path_config, args.confidence, args.nms_threshold,
                                       data.scale, data.size, args.mean, data.swapRB)
            visualizer = Visualize( reader, detector, GroundtruthReader().read(args.groundtruth_path) )
            visualizer.show()
    except Exception as e:
        raise Exception(e)

if __name__ == '__main__':
    main()