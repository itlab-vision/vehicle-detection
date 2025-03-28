import yaml

def parse_yaml_file(yaml_file):
    
    with open(yaml_file) as fh:
        parameters = yaml.safe_load(fh)
    parameters = parameters[0]
    
    mode = parameters.get('mode')

    if mode == None:
        raise ValueError('mode is not specified. This parameter is required.')
    
    if mode != 'image' and mode != 'video':
        raise ValueError('The mode is specified incorrectly.') 
    
    if mode == 'image' and parameters.get('images_path') == None:
        raise ValueError('In image mode, the images_path parameter is required.') 
    
    if mode == 'video' and parameters.get('video_path') == None:
        raise ValueError('In video mode, the video_path parameter is required.') 

    model_name = parameters.get('model_name')

    if model_name == None:
        raise ValueError('model_name is not specified. This parameter is required.')

    list_models = ['YOLOv4', 'YOLOv3_tiny', 'rcnn_resnet50', 'rcnn_resnet_v2', 'efficientdet_d1', 'efficientdet_d0', 'lite_mobilenet_v2', 'MobileNet']
    
    if not (model_name in list_models):
        raise ValueError('The model_name is specified incorrectly.\n List of acceptable models: \'YOLOv4\', \'YOLOv3_tiny\', \'rcnn_resnet50\', \'rcnn_resnet_v2\', \'efficientdet_d1\', \'efficientdet_d0\', \'lite_mobilenet_v2\', \'MobileNet\'') 

    if parameters.get('path_classes') == None:
        raise ValueError('path_classes is not specified. This parameter is required.')
    
    if parameters.get('path_weights') == None:
        raise ValueError('path_weights is not specified. This parameter is required.')
    
    if parameters.get('path_config') == None:
        parameters.update({'path_config' : None}) 
    
    if parameters.get('confidence') == None:
        parameters.update({'confidence' : 0.3})
    else:
        parameters['confidence'] = float(parameters['confidence'])
        
    if parameters.get('nms_threshold') == None:
        parameters.update({'nms_threshold' : 0.4})
    else:
        parameters['nms_threshold'] = float(parameters['nms_threshold'])
        
    if parameters.get('scale') == None:
        parameters.update({'scale' : 1.0})
    else:
        parameters['scale'] = float(parameters['scale'])
        
    if parameters.get('size') == None:
        raise ValueError('size is not specified. This parameter is required.')
    else:
        parameters['size'] = list(map(int, parameters['size'].split(' ')))
    
    if parameters.get('mean') == None:
        parameters.update({'mean' : [0.0, 0.0, 0.0]})
    else:
        parameters['mean'] = list(map(float, parameters['mean'].split(' ')))

    if parameters.get('swapRB') == None:
        parameters.update({'swapRB' : False})
    else:
        parameters['swapRB'] = bool(parameters['swapRB'])
    
    if parameters.get('groundtruth_path') == None:
        parameters.update({'groundtruth_path' : None})
        
    if parameters.get('write_path') == None:
        parameters.update({'write_path' : None})
        
    if parameters.get('silent_mode') == None:
        parameters.update({'silent_mode' : False})
    else:
        parameters['silent_mode'] = bool(parameters['silent_mode'])
    
    list_arg = ['mode', 'image', 'video', 'images_path', 'video_path', 'model_name', 'path_classes', 'path_weights', 'path_config', 'confidence',
                  'nms_threshold', 'scale', 'size', 'mean', 'swapRB', 'groundtruth_path', 'write_path', 'silent_mode']

    entered_arg = parameters.keys()
    
    for arg in entered_arg:
        if not (arg in list_arg):
            raise ValueError(f'Incorrect parameter entered: {arg}')

    return parameters