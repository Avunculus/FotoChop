from constants import *

def scaledown_fit_view(image:np.ndarray) -> np.ndarray:
    h, w  = image.shape[:2]
    shape = (round(SIDE * w / h), SIDE) if h > w else (SIDE, round(SIDE * h / w))
    image = cv.resize(image, shape)
    return image

def draw_main_win(image:np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h > SIDE or w > SIDE:
        image = scaledown_fit_view(image)
        h, w = image.shape[:2]
    # padding
    v_border = (SIDE - h) // 2
    h_border = (SIDE - w) // 2
    image = cv.copyMakeBorder(image, v_border, v_border, h_border, h_border,
                              cv.BORDER_CONSTANT, value=COLORS[-2])
    image = cv.copyMakeBorder(image, 0, 0, BAR, 0,
                              cv.BORDER_CONSTANT, value=COLORS[-1])
    return image
# MAIN = np.ones((1000, 1250, 4))
# MAIN[:, :250, :3] *= [0, 255, 0]
# MAIN[..., 3] *= 255
# cv.namedWindow('MAIN', flags=cv.WINDOW_NORMAL)
# cv.imshow('MAIN', MAIN)
# print(f'{cv.getWindowProperty('MAIN', cv.WND_PROP_TOPMOST)=}')
# cv.waitKey(0)
# cv.setWindowProperty('MAIN', cv.WND_PROP_TOPMOST, 1)
# print(f'{cv.getWindowProperty('MAIN', cv.WND_PROP_TOPMOST)=}')
# cv.setWindowProperty('MAIN', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
# cv.waitKey(0)
