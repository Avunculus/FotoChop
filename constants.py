import cv2 as cv
import numpy as np

SIDE = 720
BAR = 140
PANEL_H = 60

COLORS = [[  0,   0,   0],
          [255, 255, 255],
          [138, 138, 138],
          [222, 222,   0],
          [245,   5,  20],
          [ 10, 255,   0],
          [ 25,  30, 255],
          [ 64,  24, 124],
          [191,  28,  98]]





###################################################################################################
###################################################################################################

BUTTONS = {
    'MAIN':      {( 20,  10, 100,  50): ('SEGMENTOR', COLORS[-3]),
                  ( 20,  80, 100,  50): ('RENDER', COLORS[-4])},
    'SEGMENTOR': {( 32,  32,  64,  64): ('SAMPLE', COLORS[-2])}
    }

ASSETS = {'on': cv.imread('assets/on.png'),  # all 25 x 30 except up/down
          'off': cv.imread('assets/off.png'),       
          'trash': cv.imread('assets/trash.png'),
          'up': cv.imread('assets/up.png'),             # 25 x 20
          'down': cv.imread('assets/down.png'),         # 25 x 20
          'src color': cv.imread('assets/image.png'),
          'src gray': cv.imread('assets/image.png', cv.IMREAD_GRAYSCALE)}


SEG_PANEL = {0: {'trash' : (  5, 195, 25, 30),
                 'render': ( 40, 195, 25, 30),
                 'source': ( 75, 195, 25, 30),
                 'up'    : (110, 190, 25, 20),
                 'down'  : (110, 210, 25, 20)}}
for i in range(1, 9):
    SEG_PANEL[i] = {} 
    for btn, (x, y, w, h) in SEG_PANEL[0].items():
        SEG_PANEL[i][btn] = (x, y + PANEL_H * i, w, h)


