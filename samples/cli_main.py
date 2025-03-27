"""
CLI application "Vehicle detector"
"""
import argparse
import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.gui_application.visualizer import Visualize
from src.utils import data_reader as dr
from src.utils.frame_data_reader import FrameDataReader
from src.utils.writer import Writer
from src.vehicle_detector.detector import Detector

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-y', '--yaml',
                        type=str,
                        help = 'Path to a yaml file',
                        dest='yaml_file',
                        required=True)

    args = parser.parse_args()
    return args

def parse_yaml_file(yaml_file):
    
    with open(yaml_file) as fh:
        data = yaml.safe_load(fh)
    data = data[0]
    
    mode = data.get('mode')

    if mode == None:
        raise ValueError('mode is not specified. This parameter is required.')
    
    if mode != 'image' and mode != 'video':
        raise ValueError('The mode is specified incorrectly.') 
    
    if mode == 'image' and data.get('images_path') == None:
        raise ValueError('In image mode, the images_path parameter is required.') 
    
    if mode == 'video' and data.get('video_path') == None:
        raise ValueError('In video mode, the video_path parameter is required.') 

    model_name = data.get('model_name')

    if model_name == None:
        raise ValueError('model_name is not specified. This parameter is required.')

    list_models = ['YOLOv4', 'YOLOv3_tiny', 'rcnn_resnet50', 'rcnn_resnet_v2', 'efficientdet_d1', 'efficientdet_d0', 'lite_mobilenet_v2', 'MobileNet']
    
    if not (model_name in list_models):
         raise ValueError('The model_name is specified incorrectly.\n List of acceptable models: \'YOLOv4\', \'YOLOv3_tiny\', \'rcnn_resnet50\', \'rcnn_resnet_v2\', \'efficientdet_d1\', \'efficientdet_d0\', \'lite_mobilenet_v2\', \'MobileNet\'') 

    if data.get('path_classes') == None:
         raise ValueError('path_classes is not specified. This parameter is required.')
    
    if data.get('path_weights') == None:
         raise ValueError('path_weights is not specified. This parameter is required.')
    
    if data.get('path_config') == None:
         raise ValueError('path_config is not specified. This parameter is required.')
    
    if data.get('confidence') == None:
        data.update({'confidence' : 0.3})
    else:
        data['confidence'] = float(data['confidence'])
        
    if data.get('nms_threshold') == None:
        data.update({'nms_threshold' : 0.4})
    else:
        data['nms_threshold'] = float(data['nms_threshold'])
        
    if data.get('scale') == None:
        data.update({'scale' : 1.0})
    else:
        data['scale'] = float(data['scale'])
        
    if data.get('size') == None:
         raise ValueError('size is not specified. This parameter is required.')
    else:
        data['size'] = list(map(int, data['size'].split(' ')))
    
    if data.get('mean') == None:
        data.update({'mean' : [0.0, 0.0, 0.0]})
    else:
        data['mean'] = list(map(float, data['mean'].split(' ')))

    if data.get('swapRB') == None:
        data.update({'swapRB' : False})
    else:
        data['swapRB'] = bool(data['swapRB'])
    
    if data.get('groundtruth_path') == None:
        data.update({'groundtruth_path' : None})
        
    if data.get('write_path') == None:
        data.update({'write_path' : None})
    
    list_arg = ['mode', 'image', 'video', 'images_path', 'video_path', 'model_name', 'path_classes', 'path_weights', 'path_config', 'confidence',
                  'nms_threshold', 'scale', 'size', 'mean', 'swapRB', 'groundtruth_path', 'write_path']

    entered_arg = data.keys()
    
    for arg in entered_arg:
        if not (arg in list_arg):
            raise ValueError(f'Incorrect parameter entered: {arg}')

    return data

def main():
    """
    Main execution function for the visualizer application.

    Initializes data reader, detector, and visualizer components based on CLI arguments.
    Shows visualization using the following workflow:
    
        1. Creates FrameDataReader based on input mode (video/image)
        2. Initializes a writer with 'write_path' implementation
        2. Initializes a detector with 'fake' implementation
        3. Loads groundtruth data if provided
        4. Configures visualizer with reader, detector, and groundtruth
        5. Starts visualization display

    Requires:
        - Either video_path or images_path argument must match the specified mode
        - model_path must point to a valid model file
    """
    try:
        args = cli_argument_parser()
        data = parse_yaml_file(args.yaml_file)
            
        reader = 0
        if data['mode'] == 'image':
            reader = FrameDataReader.create(data['mode'], data['images_path'])
        elif data['mode'] == 'video':
            reader = FrameDataReader.create(data['mode'], data['video_path'])
            
        writer = Writer.create(data['write_path'])
        detector = Detector.create(data['model_name'], data['path_classes'], data['path_weights'], data['path_config'], data['confidence'],
                                   data['nms_threshold'], data['scale'], data['size'], data['mean'], data['swapRB'])
        gtreader = dr.FakeGTReader(data['groundtruth_path'])
        visualizer = Visualize(reader, writer, detector, gtreader.read())
        visualizer.show()

    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()