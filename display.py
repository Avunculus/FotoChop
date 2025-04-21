from constants import *

# BUTTON = np.ix_(np.arange(10, 190), np.arange(10, 190))

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
                              cv.BORDER_CONSTANT, value=COLORS[-3])
    image = cv.copyMakeBorder(image, 0, 0, BAR, 0,
                              cv.BORDER_CONSTANT, value=COLORS[-2])
    for (x, y, w, h), (name, color) in BUTTONS['MAIN'].items():
        ix = np.ix_(np.arange(y, y + h), np.arange(x, x + w))
        image[ix] = color
    # image[BUTTON] = COLORS[4]
    return image

def draw_chopjob_win(image:np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h > SIDE or w > SIDE:
        image = scaledown_fit_view(image)
        h, w = image.shape[:2]
    # padding
    v_border = (SIDE - h) // 2
    h_border = (SIDE - w) // 2
    image = cv.copyMakeBorder(image, v_border, v_border, h_border, h_border,
                              cv.BORDER_CONSTANT, value=COLORS[2])
    image = cv.copyMakeBorder(image, 0, 0, BAR, 0,
                              cv.BORDER_CONSTANT, value=COLORS[5])
    for (x, y, w, h), (name, color) in BUTTONS['CHOPJOB'].items():
        ix = np.ix_(np.arange(y, y + h), np.arange(x, x + w))
        image[ix] = color
    return image

def check_click(pos:tuple[int,int], buttons:dict[tuple,str]) -> str:
    for (x, y, w, h), name in buttons.items():
        if pos[0] in range(x, x + w) and pos[1] in range(y, y + h):
            return name
    return ''
