import csv
import random
class GroundtruthReader():
    @staticmethod
    def read(file_path):
        """
        Parsing CSV file with groundtruths.

        :param file_path: The path to the file with groundtruths.
        :return: list[tuples] of parsed data by rows.
        """
        parsed_data = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
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
            print(f"File {file_path} was not found.")
        except Exception as e:
            print(f"Error when reading the file {file_path}: {e}")

        return parsed_data

class FakeGroundtruthReader(GroundtruthReader):
    def __init__(self,
                max_frames = 1000,
                obj_classes = None,
                img_size = (1920, 1080),
                seed = None):
        """
        Initializing the synthetic data generator
        
        :param max_frames: The maximum number of frames to generate
        :param obj_classes:  List of object classes
        :param img_size: Image size (width, height)
        :param seed: Seed for reproducibility
        """
        super().__init__()
        self.max_frames = max_frames
        self.obj_classes = obj_classes or ['car', 'truck', 'bus']
        self.img_width, self.img_height = img_size
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def read(self, file_path = None):
        """
        Parsing CSV file with groundtruths.

        :param file_path: The path to the file with groundtruths.
        :return: list[tuples] of parsed data by rows.
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
        """Генерация случайного bounding box."""
        x = int(random.uniform(0, self.img_width - 50))
        y = int(random.uniform(0, self.img_height - 50))
        w = int(random.uniform(50, self.img_width - x))
        h = int(random.uniform(50, self.img_height - y))
        return (x, y, h, w)

class DetectionReader:
    @staticmethod
    def read(file_path):
        """
        Parsing CSV file with detections.

        :param file_path: The path to the file with detections.
        :return: list[tuples] of parsed data by rows.
        """
        parsed_data = []
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