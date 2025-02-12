import cv2 as cv
import os
from abc import ABC, abstractmethod

class DataReader(ABC):
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @staticmethod
    def create(args):
        if args.mode == 'video':
            return VideoDataReader(args)
        elif args.mode == 'image':
            return ImgDataReader(args)
        else:
            raise ValueError(f'Unsupported mode: {arg.mode}')

class VideoDataReader(DataReader):

    def __init__(self, paths):
        self.video_path = paths.video_path
        self.cap = cv.VideoCapture(paths.video_path)
        if not self.cap.isOpened():
            raise ValueError(f'Cannot open video file: {paths.video_path}')
        if (paths.groundtruth_path):
            self.groundtruth_path = paths.groundtruth_path

    def __iter__(self):
        return self

    def __next__(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                self.cap.release()
                raise StopIteration
        else:
            raise StopIteration

class ImgDataReader(DataReader):
    def __init__(self, paths):
        self.index = 0
        self.directory_path = paths.images_path
        if not os.path.exists(paths.images_path):
            raise ValueError(f'Directory does not exist: {paths.images_path}')
        self.image_files = [
            os.path.join(paths.images_path, f) for f in os.listdir(paths.images_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
        if (paths.groundtruth_path):
            self.groundtruth_path = paths.groundtruth_path
        

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.image_files):
            img_path = self.image_files[self.index]
            self.index += 1
            img = cv.imread(img_path)
            if img is None:
                raise ValueError(f'Cannot read image file: {img_path}')
            return img
        else:
            raise StopIteration