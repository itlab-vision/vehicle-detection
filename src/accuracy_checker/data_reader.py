import csv


class GroundtruthReader:
    @staticmethod
    def read_key_class_name(file_path):
        """
        Парсинг CSV-файла с истинной разметкой.

        :param file_path: Путь к файлу с разметкой.
        :return: Словарь {class_name: {frame_id: [список ограничивающих прямоугольников]}}.
        """
        annotations = {}
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) != 6:
                        print(f"Некорректная строка в файле: {row}")
                        continue

                    frame_id, class_name, x1, y1, x2, y2 = row
                    frame_id = int(frame_id)
                    bbox = [float(x1), float(y1), float(x2), float(y2)]

                    if class_name not in annotations:
                        annotations[class_name] = {}
                    if frame_id not in annotations[class_name]:
                        annotations[class_name][frame_id] = []
                    annotations[class_name][frame_id].append(bbox)

        except FileNotFoundError:
            print(f"Файл {file_path} не найден.")
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}: {e}")

        return annotations

    @staticmethod
    def read_key_frame_id(file_path):
        """
        Парсинг CSV-файла с истинной разметкой.

        :param file_path: Путь к файлу с разметкой.
        :return: Словарь {frame_id: {class_name: [список ограничивающих прямоугольников]}}.
        """
        annotations = {}
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) != 6:
                        print(f"Некорректная строка в файле: {row}")
                        continue

                    frame_id, class_name, x1, y1, x2, y2 = row
                    frame_id = int(frame_id)
                    bbox = [float(x1), float(y1), float(x2), float(y2)]

                    if frame_id not in annotations:
                        annotations[frame_id] = {}
                    if class_name not in annotations[frame_id]:
                        annotations[frame_id][class_name] = []
                    annotations[frame_id][class_name].append(bbox)

        except FileNotFoundError:
            print(f"Файл {file_path} не найден.")
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}: {e}")

        return annotations


class DetectionReader:
    @staticmethod
    def read_key_class_name(file_path):
        """
        Парсинг CSV-файла с предсказаниями детектора.

        :param file_path: Путь к файлу с разметкой.
        :return: Словарь {class_name: {frame_id: [список ограничивающих прямоугольников]}}.
        """
        annotations = {}
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) != 7:
                        print(f"Некорректная строка в файле: {row}")
                        continue

                    frame_id, class_name, x1, y1, x2, y2, confidence = row
                    frame_id = int(frame_id)
                    confidence = float(confidence)
                    bbox = [float(x1), float(y1), float(x2), float(y2), confidence]

                    if class_name not in annotations:
                        annotations[class_name] = {}
                    if frame_id not in annotations[class_name]:
                        annotations[class_name][frame_id] = []
                    annotations[class_name][frame_id].append(bbox)

        except FileNotFoundError:
            print(f"Файл {file_path} не найден.")
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}: {e}")

        return annotations

    @staticmethod
    def read_key_frame_id(file_path):
        """
        Парсинг CSV-файла с предсказаниями детектора.

        :param file_path: Путь к файлу с разметкой.
        :return: Словарь {frame_id: {class_name: [список ограничивающих прямоугольников]}}.
        """
        annotations = {}
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) != 7:
                        print(f"Некорректная строка в файле: {row}")
                        continue

                    frame_id, class_name, x1, y1, x2, y2, confidence = row
                    frame_id = int(frame_id)
                    confidence = float(confidence)
                    bbox = [float(x1), float(y1), float(x2), float(y2), confidence]

                    if frame_id not in annotations:
                        annotations[frame_id] = {}
                    if class_name not in annotations[frame_id]:
                        annotations[frame_id][class_name] = []
                    annotations[frame_id][class_name].append(bbox)

        except FileNotFoundError:
            print(f"Файл {file_path} не найден.")
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}: {e}")

        return annotations
