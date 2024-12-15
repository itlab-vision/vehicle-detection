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
        # Проверяем, чтобы метод вызывался только на уровне базового класса
        if DataReader != globals().get('DataReader'):
            raise TypeError("Метод 'create' может быть вызван только в классе DataReader.")
        if args.mode == "video":
            return VideoDataReader(args)
        elif args.mode == "image":
            return ImgDataReader(args)
        else:
            raise ValueError(f"Неподдерживаемый режим: {mode}")

class VideoDataReader(DataReader):

    def __init__(self, args):
        self.video_path = args.video_path
        self.cap = cv.VideoCapture(args.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Не получилось считать видео: {args.video_path}")

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

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

class ImgDataReader(DataReader):
    def __init__(self, args):
        self.index = 0
        self.directory_path = args.images_path
        if not os.path.exists(args.images_path):
            raise ValueError(f"Данной директории не существует: {args.images_path}")
        self.image_files = [
            os.path.join(args.images_path, f) for f in os.listdir(args.images_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
        # self.image_files.sort()  # Для предсказуемого порядка
        

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.image_files):
            img_path = self.image_files[self.index]
            self.index += 1
            img = cv.imread(img_path)
            if img is None:
                raise ValueError(f"Не получилось считать изображение: {img_path}")
            return img
        else:
            raise StopIteration
    def print_path(self):
        print(self.image_files)
