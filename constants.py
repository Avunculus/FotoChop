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

BUTTONS = {
    'MAIN'    : {( 20,  10, 100,  50): ('SEGMENT', COLORS[3]),
                 ( 20,  80, 100,  50): ('RENDER' , COLORS[4])},
    'SEGMENT' : {( 32,  32,  64,  64): ('CHOP IT', COLORS[-2]),
                 ( 32,  32,  64,  64): ('UNDO'   , COLORS[-2]),
                 ( 32,  32,  64,  64): ('CLEAR'  , COLORS[-2]),
                 ( 32,  32,  64,  64): ('ACCEPT' , COLORS[-2])},
    'FINISHER': {( 32,  32,  64,  64): ('ERODE'  , COLORS[-2]),
                 ( 32,  32,  64,  64): ('DILATE' , COLORS[-2]),
                 ( 32,  32,  64,  64): ('APPLY'  , COLORS[-2]),
                 ( 32,  32,  64,  64): ('FINISH' , COLORS[-2])}
    }

SEG_RECT = (0, 180, 140, 60)

SEG_BTNS = {
    'trash' : ((  5, 195, 25, 30), [cv.imread('assets/trash.png')]),
    'render': (( 40, 195, 25, 30), [cv.imread('assets/off.png'), 
                                    cv.imread('assets/on.png')]),
    'source': (( 75, 195, 25, 30), [np.ones((30, 25, 3)) * c for c in COLORS] +\
                                   [cv.imread('assets/image.png'),
                                    cv.imread('assets/image.png', cv.IMREAD_GRAYSCALE)]),
    'up'    : ((110, 190, 25, 20), [cv.imread('assets/up.png', cv.IMREAD_GRAYSCALE),
                                    cv.imread('assets/up.png')]),
    'down'  : ((110, 210, 25, 20), [cv.imread('assets/down.png', cv.IMREAD_GRAYSCALE),
                                    cv.imread('assets/down.png')])
    }
# Segment.button_states == ordered indeces for btn states:
#   cycle?   y  y  n  n
#   btn:  t [r  s  u  d]
# range:  1 [2  11 2  2]
#   def:  0 [0, 9, 0, 0]

###################################################################################################
###################################################################################################

# IMAGES = {'on': cv.imread('assets/on.png'),  # all 25 x 30 except up/down
#           'off': cv.imread('assets/off.png'),       
#           'trash': cv.imread('assets/trash.png'),
#           'up': cv.imread('assets/up.png'),             # 25 x 20
#           'down': cv.imread('assets/down.png'),         # 25 x 20
#           'src color': cv.imread('assets/image.png'),
#           'src gray': cv.imread('assets/image.png', cv.IMREAD_GRAYSCALE)}


# SEG_PANEL = {0: {'trash' : (  5, 195, 25, 30),
#                  'render': ( 40, 195, 25, 30),
#                  'source': ( 75, 195, 25, 30),
#                  'up'    : (110, 190, 25, 20),
#                  'down'  : (110, 210, 25, 20)}}
# for i in range(1, 9):
#     SEG_PANEL[i] = {} 
#     for btn, (x, y, w, h) in SEG_PANEL[0].items():
#         SEG_PANEL[i][btn] = (x, y + PANEL_H * i, w, h)


