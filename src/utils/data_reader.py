"""
Ground Truth and Detection Data Readers Module

Provides abstract and concrete implementations for reading annotation data from:
- CSV files with ground truth
- CSV files with detection results
- Synthetic data generation for testing

Classes:
    GroundtruthReader: Abstract base class for data readers
    CsvGTReader: CSV parser for ground truth annotations
    FakeGTReader: Synthetic data generator for testing
    DetectionReader: CSV parser for detection results with confidence scores

Dependencies:
    - csv module for file parsing
    - random module for synthetic data generation
"""
import csv
import random
from abc import ABC, abstractmethod

class GroundtruthReader(ABC):
    """Abstract base class for data reading implementations.
    
    Defines common interface for reading annotation data from different sources.
    
    Args:
        filepath (str): Path to data source (file or dummy source)
    
    Methods:
        read: Abstract method to be implemented in subclasses
    """

    def __init__(self, filepath):
        self.file_path = filepath

    @abstractmethod
    def read(self):
        """Parse and return annotation data.
        
        Returns:
            list: Annotation data in format specific to concrete implementation
        """
        pass

class CsvGTReader(GroundtruthReader):
    def __init__(self, filepath):
        super().__init__(filepath)

    def read(self):
        """
        Parsing CSV file with groundtruths.

        :param file_path: The path to the file with groundtruths.
        :return: list[tuples] of parsed data by rows.
        """
        parsed_data = []
        try:
            with open(self.file_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) != 6:
                        print(f"Incorrect line in the file: {row}")
                        continue

                    frame_id, class_name, x1, y1, x2, y2 = row
                    row_data = (int(frame_id), str(class_name), int(x1), int(y1),
                                int(x2), int(y2))
                    parsed_data.append(row_data)

        except FileNotFoundError:
            print(f"File {self.file_path} was not found.")
        except Exception as e:
            print(f"Error when reading the file {self.file_path}: {e}")

        return parsed_data

class FakeGTReader(GroundtruthReader):
    """Synthetic ground truth data generator for testing.
    
    Generates random annotations with:
    - Configurable number of frames
    - Random object classes
    - Valid bounding boxes within image dimensions
    - Reproducible results through seeding
    
    Args:
        file_path (str): Dummy parameter for interface compatibility
    """
    def __init__(self, file_path):
        super().__init__(file_path)
        self.max_frames = 1000
        self.obj_classes = ['car', 'truck', 'bus']
        self.img_width, self.img_height = (1920, 1080)
        self.seed = 1
        random.seed(self.seed)

    def read(self):
        """Generate synthetic annotation data.
        
        Returns:
            list[tuple]: Generated data in format 
            (frame_id, class_name, x1, y1, x2, y2)
            
        Notes:
            - 20% chance to skip frames randomly
            - 1-5 objects per frame
            - Coordinates rounded to 2 decimal places
        """
        data = []
        num_frames = random.randint(self.max_frames // 2, self.max_frames)
        for frame_id in range(num_frames):
            if random.random() < 0.2:
                continue
            num_objects = random.randint(1, 5)
            for _ in range(num_objects):
                x1, y1, w, h = self.__generate_bbox()
                x2 = x1 + w
                y2 = y1 + h
                
                data.append((
                    frame_id,
                    random.choice(self.obj_classes),
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ))
        return data
    
    def __generate_bbox(self):
        """Generate random valid bounding box coordinates.
        
        Returns:
            tuple: (x, y, width, height) coordinates within image bounds
        """
        x = int(random.uniform(0, self.img_width - 50))
        y = int(random.uniform(0, self.img_height - 50))
        w = int(random.uniform(50, self.img_width - x))
        h = int(random.uniform(50, self.img_height - y))
        return (x, y, h, w)

class DetectionReader(GroundtruthReader):
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        """
        Parsing CSV file with detections.

        :param file_path: The path to the file with detections.
        :return: list[tuples] of parsed data by rows.
        """
        parsed_data = []
        try:
            with open(self.file_path, mode='r', encoding='utf-8') as file:
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
            print(f"File {self.file_path} was not found.")
        except Exception as e:
            print(f"Error when reading the file {self.file_path}: {e}")
        return parsed_data