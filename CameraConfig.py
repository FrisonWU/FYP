import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('key template.jpg',cv2.IMREAD_GRAYSCALE)
    size = img.shape
    print(size)