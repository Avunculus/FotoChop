from constants import *

# BUTTON = np.ix_(np.arange(10, 190), np.arange(10, 190))

def square_frame(image:np.ndarray, side:int):
    """returns image resized to fit long side into square frame, with padding (return image is a square)"""
    h, w  = image.shape[:2]
    shape = (round(side * w / h), side) if h > w else (side, round(side * h / w))
    square = cv.resize(image, shape)
    if h > w: # portrait
        cv.copyMakeBorder(square, 0, 0, (side - w) // 2, (side - w) // 2, cv.BORDER_CONSTANT, value=0) #[0] * image.shape[2])
    else:
        cv.copyMakeBorder(square, (side - h) // 2, (side - h) // 2, 0, 0, cv.BORDER_CONSTANT, value=0) 
    return square


def scaledown_fit_view(image:np.ndarray, side=SIDE) -> np.ndarray:
    h, w  = image.shape[:2]
    shape = (round(side * w / h), side) if h > w else (side, round(side * h / w))
    image = cv.resize(image, shape)
    return image

def draw_main_buttons(win:np.ndarray, selection:str=''):
    for (x, y, w, h), (name, color) in BUTTONS['MAIN'].items():
        ix = np.ix_(np.arange(y, y + h), np.arange(x, x + w))
        win[ix] = color
        if name.isdecimal(): # segment select
            cv.rectangle(win, (x, y), (x + w, y + h), COLORS[2], 6)
        if name == selection:
            cv.rectangle(win, (x, y), (x + w, y + h), COLORS[5], 6)

def draw_main_win(image:np.ndarray) -> np.ndarray:
    # h, w = image.shape[:2]
    # if h > SIDE or w > SIDE:
    image = scaledown_fit_view(image)
    h, w = image.shape[:2]
    # padding
    v_border = (SIDE - h) // 2
    h_border = (SIDE - w) // 2
    # img_pos = (BAR + h_border, v_border, w, h)
    image = cv.copyMakeBorder(image, v_border, v_border, h_border, h_border,
                              cv.BORDER_CONSTANT, value=COLORS[-3])
    image = cv.copyMakeBorder(image, 0, 0, BAR, 0,
                              cv.BORDER_CONSTANT, value=COLORS[-2])
    draw_main_buttons(image)
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

# def hilite_segment(segment:np.ndarray, source:np.ndarray, win:np.ndarray):
