import cv2 as cv
from datareader import DataReader

class PseudoDetector:
    def __init__(self, annotation_file):
        
        frame_annotations = {}
        with open(annotation_file, 'r') as file:
            annotations = file.readlines()
        
        for line in annotations:
            parts = line.strip().split()
            frame_idx = int(parts[0])
            label = parts[1]
            x1, y1 = int(parts[2]), int(parts[3])
            x2, y2 = int(parts[4]), int(parts[5])
            
            if frame_idx not in frame_annotations:
                frame_annotations[frame_idx] = []
            frame_annotations[frame_idx].append((label, x1, y1, x2, y2))
        self.frame_annotations = frame_annotations
    def getLayouts(self, ind):
        return self.frame_annotations.get(ind, [])

class Visualize:
    def __init__(self, datareader:DataReader, detector:PseudoDetector):
        self.datareader = datareader
        self.detector = detector
    def show(self):
        # метод детектора который вернёт разметку?
        # Или же я её получить должен?
        ind = 0
        try:
            for image in self.datareader:
                if image is not None:
                    
                    for box in self.detector.getLayouts(ind):
                        label, x1, y1, x2, y2 = box
                        # Рисуем прямоугольник
                        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Зеленый прямоугольник, толщина 2
                        # Подписываем класс
                        cv.putText(image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv.imshow("Image", image)
                    ind+=1
                else:
                    break
                if cv.waitKey(25) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Возникла ошибка: {e}")
        finally:
            cv.destroyAllWindows()

