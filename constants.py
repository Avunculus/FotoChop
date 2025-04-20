import cv2 as cv
import numpy as np

SIDE = 960

COLORS = [[  0,   0,   0],
          [255, 255, 255],
          [138, 138, 138],
          [245,   5,  20],
          [ 10, 255,   0],
          [ 25,  30, 255],
          [ 64,  24, 124]]

def resize_fit_view(image:np.ndarray) -> np.ndarray:
    h, w  = image.shape[:2]
    shape = (round(SIDE * w / h), SIDE) if h > w else (SIDE, round(SIDE * h / w))
    img_view = cv.resize(image, shape)
    h, w = img_view.shape[:2]
    win = np.ones(shape=(SIDE, SIDE, 3)) * COLORS[5]
    dh, dw = (SIDE - h, SIDE - w)

    win[dh // 2: dh // 2 + h, dw // 2: dw // 2 + w, :] = img_view
    return win
