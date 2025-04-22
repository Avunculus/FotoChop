import os
from display import *

def slice_rect(image:np.ndarray, rect:tuple) -> np.ndarray:
    x, y, w, h = rect
    return image[y: y + h, x: x + w, :]

def pick_source() -> str: #-> tuple[np.ndarray, str]:
    """defines gloabal: SOURCE"""
    ###
    fn = 'uncle baby billy.jpg'
    ###
    name = 'sources/' + fn
    image = cv.imread(name)
    match image.shape[2]:   # set color format to 3-CHANNEL
        case 1: image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)  # grayscale
        case 4: image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)  # 4-channel
    global SOURCE
    SOURCE = image.copy()
    return 'sources/' + fn.split('.')[0] + '/'  # (image, 'sources/' + fn.split('.')[0] + '/')

def get_roi(image:np.ndarray) -> tuple[int,int,int,int]:
    """integer scaledown image, scales ROI rect back up to original"""
    # scaled = square_frame(image, SIDE, pad=False)
    scaled, factor = integer_scaledown(image)
    roi = (0, 0, 0, 0)
    # while any([i == 0 for i in roi[2:]]):
    roi = cv.selectROI('select ROI, then press spacebar', scaled)
    cv.destroyWindow('select ROI, then press spacebar')
    roi = tuple([i * factor for i in roi])
    return roi

class Segmentor:
    def __init__(self):
        self.window = np.zeros((SIDE, 2 * SIDE + BAR, 3))
        # draw sidebar ui
    def handle_mouse(self, event:int, x:int, y:int, flags:int, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(f'clicked \'SEGMENT\' window @ ({x}, {y})')
    def get_source(self) -> np.ndarray:
        return slice_rect(SOURCE, self.rect)
    def run(self):
        self.rect = get_roi(SOURCE) # !! getting zero-dim error for w|h...
        ... # check dims for 0-width|hgt? *** select_roi needs fixing. Thread issue.
        self.source = self.get_source()
        cv.namedWindow('SEGMENTOR')
        cv.setMouseCallback('SEGMENTOR', self.handle_mouse)
        cv.imshow('SEGMENTOR', self.window)

class ChopJob:
    def __init__(self, segments: list[np.ndarray]):
        self.window = np.zeros((SIDE, 2 * SIDE + BAR, 3))
        # draw sidebar ui
    def handle_mouse(self, event:int, x:int, y:int, flags:int, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(f'clicked \'CHOPJOB\' window @ ({x}, {y})')
    def run(self):
        ...
        cv.namedWindow('CHOPJOB')
        cv.setMouseCallback('CHOPJOB', self.handle_mouse)
        cv.imshow('CHOPJOB', self.window)

def mouse_main(event:int, x:int, y:int, flags:int, param):
    # source, win_main, img_pos = param
    if event == cv.EVENT_LBUTTONDOWN:
        button = check_click((x, y), {rect:val[0] for rect, val in BUTTONS['MAIN'].items()})
        if button == 'SEGMENTOR':
            Segmentor().run()
        elif button == 'CHOPJOB':
            ChopJob(SEGMENTS.copy()).run()

def main(path:str) -> bool:
    win_main = draw_main_win(SOURCE)
    global SEGMENTS
    SEGMENTS = []
    for seg in os.listdir(path):
        bitmask = cv.imread(path + seg) // 255
        SEGMENTS.append(bitmask, )
    print(f'read {len(SEGMENTS)} segment masks from file.')
    if len(SEGMENTS) > 9:
        print(f'warning! too many segment masks ({len(SEGMENTS)}) loaded.')
        SEGMENTS = SEGMENTS[:9]
    cv.namedWindow('MAIN')
    cv.setMouseCallback('MAIN', mouse_main)
    while True:
        cv.imshow('MAIN', win_main)
        key = cv.waitKey(1)
        if key == 27:
            break
        elif key == 18:  # ctrl-r -> restart
            cv.destroyAllWindows()
            return True
        elif key > 0:
            print(f'{key=}')
    cv.destroyAllWindows()
    return False

if __name__ == '__main__':
    path = pick_source()
    repeat = main(path)
    while repeat:
        path = pick_source()
        repeat = main(path)
    print('Done.')