import cv2 as cv
import os
from abc import ABC, abstractmethod

class FrameDataReader(ABC):
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @staticmethod
    def create(mode, dir_path):
        if mode == "video":
            return VideoDataReader(dir_path)
        elif mode == "image":
            return ImgDataReader(dir_path)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

class VideoDataReader(FrameDataReader):

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        

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

class ImgDataReader(FrameDataReader):
    def __init__(self, dir_path):
        self.index = 0
        self.directory_path = dir_path
        if not os.path.exists(dir_path):
            raise ValueError(f"Directory does not exist: {dir_path}")
        self.image_files = [
            os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ] 

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.image_files):
            img_path = self.image_files[self.index]
            self.index += 1
            img = cv.imread(img_path)
            if img is None:
                raise ValueError(f"Cannot read image file: {img_path}")
            return img
        else:
            raise StopIteration