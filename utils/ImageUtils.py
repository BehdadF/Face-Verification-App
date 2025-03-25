import cv2
import time

class ImageUtils:
    @staticmethod
    def openImage(img_path):
        """
            Opens an image using cv2 for a numpy represenation

            ARGS:
                - img_path: Path to image to be opened

            Returns:
                - numpy array representation of the image
        """
        return cv2.imread(img_path)

    @staticmethod
    def bgrToRgb(img):
        """
            Opens an image using cv2 for a numpy represenation

            ARGS:
                - img: numpy representation of the image

            Returns:
                - bgr to rgb converted array
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def captureImg(device=0):
        try:
            cap = cv2.VideoCapture(device)
        
            time.sleep(0.1)

            ret, frame = cap.read()
            cap.release()
            if ret:
                return frame
            else:
                print("Unalbe to capture the image")
        except:
            print("The specified device is not available")

    # @staticmethod
    # def 

