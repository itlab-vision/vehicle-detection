import cv2 as cv
from datareader import DataReader

class Visualize:
    def __init__(self, datareader, detector):
        self.datareader = datareader
        self.detector = detector
    def show(self):
        # метод детектора который вернёт разметку?
        # Или же я её получить должен?
        
        try:
            for image in self.datareader:
                if image is not None:
                    cv.imshow("Image", image)
                    
                else:
                    break
                if cv.waitKey(25) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Возникла ошибка: {e}")
        finally:
            cv.destroyAllWindows()

