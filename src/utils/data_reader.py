import csv
import cv2 as cv
import os
from abc import ABC, abstractmethod
import random
class DataReader(ABC):
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
        elif mode == "groundtruth":
            return GroundtruthReader(dir_path)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

class VideoDataReader(DataReader):

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

class ImgDataReader(DataReader):
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

class FakeGTReader():
    def __init__(
        self,
        total_frames: int = 1000,
        prob_missing: float = 0.5,
        image_width: int = 1920,
        image_height: int = 1080,
        max_objects_per_frame: int = 5,
        seed: int = None
    ):
        self.total_frames = total_frames
        self.prob_missing = prob_missing
        self.image_width = image_width
        self.image_height = image_height
        self.max_objects_per_frame = max_objects_per_frame
        self.labels = ['car', 'truck', 'bus']
        
        if seed is not None:
            random.seed(seed)
            
    def __generate_bbox(self):
        """Генерация случайного bounding box."""
        x = random.randint(0, self.image_width - 1)
        y = random.randint(0, self.image_height - 1)
        w = random.randint(1, self.image_width - x)
        h = random.randint(1, self.image_height - y)
        return (x, y, h, w)

    def read(self, file_path:str = None):
        parsed_data = list()
        if (file_path):
            try:
                with open(file_path, 'r') as file:
                    annotations = file.readlines()
                
                for line in annotations:
                    
                    parts = line.strip().split()
                    
                    frame_idx = int(parts[0])
                    label = parts[1]
                    x1, y1 = int(parts[2]), int(parts[3])
                    x2, y2 = int(parts[4]), int(parts[5])
                    
                    
                    row_data = (frame_idx, label, x1, y1, x2, y2)
                    parsed_data.append(row_data)
                return parsed_data
            except FileNotFoundError:
                print(f"File {file_path} was not found.")
            except Exception as e:
                print(f"Error when reading the file {file_path}: {e}")


        else:
            for frame in range(self.total_frames):
                
                if random.random() < self.prob_missing:
                    continue
                    
                
                num_objects = random.randint(0, self.max_objects_per_frame)
                for _ in range(num_objects):
                    label = random.choice(self.labels)
                    x, y, h, w = self.__generate_bbox()
                    parsed_data.append((frame, label, x, y, h, w))
            
        return parsed_data


class GroundtruthReader():
    @staticmethod
    def read(file_path):
        """
        Parsing CSV file with groundtruths.

        :param file_path: The path to the file with groundtruths.
        :return: list[tuples] of parsed data by rows.
        """
        parsed_data = list()
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) != 6:
                        print(f"Incorrect line in the file: {row}")
                        continue

                    frame_id, class_name, x1, y1, x2, y2 = row
                    row_data = (int(frame_id), str(class_name), float(x1), float(y1),
                                float(x2), float(y2))
                    parsed_data.append(row_data)

        except FileNotFoundError:
            print(f"File {file_path} was not found.")
        except Exception as e:
            print(f"Error when reading the file {file_path}: {e}")

        return parsed_data


class DetectionReader:
    @staticmethod
    def read(file_path):
        """
        Parsing CSV file with detections.

        :param file_path: The path to the file with detections.
        :return: list[tuples] of parsed data by rows.
        """
        parsed_data = list()
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) != 7:
                        print(f"Incorrect line in the file: {row}")
                        continue

                    frame_id, class_name, x1, y1, x2, y2, confidence = row
                    row_data = (int(frame_id), str(class_name), float(x1), float(y1),
                                float(x2), float(y2), float(confidence))
                    parsed_data.append(row_data)

        except FileNotFoundError:
            print(f"File {file_path} was not found.")
        except Exception as e:
            print(f"Error when reading the file {file_path}: {e}")

        return parsed_data