import cv2 as cv
import numpy as np

shape = cv.imread('sources/uncle baby billy.jpg').shape[:2]
rects = [(663, 140, 100, 190), (206, 206, 80, 80)]
for i, (x, y, w, h) in enumerate(rects):
    layer = np.zeros(shape, dtype=np.uint8)
    ix = np.ix_(np.arange(y, y + h), np.arange(x, x + w))
    layer[ix] = 255
    cv.imwrite('sources/uncle baby billy/' + repr(i) + '.jpg', layer)