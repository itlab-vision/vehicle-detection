import csv


class GroundtruthReader:
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