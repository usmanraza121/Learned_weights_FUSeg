import cv2
import numpy as np

def to_grayscale_as_y_from_yuv(path):
    img = cv2.imread(path)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)
    return y

def to_graycale(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
