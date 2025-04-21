import os
from display import *

def pick_source() -> tuple[np.ndarray, str]:
    ###
    fn = 'uncle baby billy.jpg'
    ###
    name = 'sources/' + fn
    
    image = cv.imread(name)
    match image.shape[2]: # check Color format, convert to 4 channel?
        case 1: image = cv.cvtColor(image, cv.COLOR_GRAY2BGRA) # grayscale
        case 3: image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)  # 3-channel
    global SOURCE
    SOURCE = image.copy()
    return (image, 'sources/' + fn.split('.')[0] + '/')

def get_roi(image:np.ndarray) -> tuple[int,int,int,int]:
    scaled = scaledown_fit_view(image)
    fx = scaled.shape[1] / image.shape[1]
    fy = scaled.shape[0] / image.shape[0]
    roi = (0, 0, 0, 0)
    while any([i == 0 for i in roi[2:]]):
        roi = cv.selectROI('select ROI, then press spacebar', scaled)
    cv.destroyWindow('select ROI, then press spacebar')
    x, y, w, h = roi
    roi = (round(x / fx), round(y / fy), round(w / fx), round(h / fy))
    return roi


class Segmentor:
    def __init__(self, source:np.ndarray):
        self.source = cv.cvtColor(source, cv.COLOR_BGRA2BGR) # 3-channel for grabcut
        self.window = np.zeros((SIDE, 2 * SIDE + BAR, 3))

    def handle_mouse(self, event:int, x:int, y:int, flags:int, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(f'clicked \'SEGMENT\' window @ ({x}, {y})')

    def run(self):
        # self.segment = np.zeros(self.source.shape[:2], dtype=np.uint8)
        rect = get_roi(self.source)
        x, y, w, h = rect
        self.roi = self.source[y: y + h, x: x + w, :]
        # draw window
        self.window[0:SIDE, 0:BAR, :] = [255, 0, 128]
        roi_view = scaledown_fit_view(self.roi)
        cut_view = np.zeros_like(roi_view)
        h, w = roi_view.shape[:2]
        v_border = (SIDE - h) // 2
        h_border = (SIDE - w) // 2
        roi_rect = (BAR + h_border, v_border, w, h)
        x, y, w, h = roi_rect
        self.window[y: y + h, x: x + w, :] = roi_view
        cut_rect = (SIDE + BAR + 3 * h_border, v_border, w, h)
        x, y, w, h = cut_rect
        self.window[y: y + h, x: x + w, :] = cut_view
        cv.namedWindow('SEGMENTOR')
        cv.setMouseCallback('SEGMENTOR', self.handle_mouse)
        cv.imshow('SEGMENTOR', self.window)


def mouse_main(event:int, x:int, y:int, flags:int, param):
    source, win_main, img_pos = param
    if event == cv.EVENT_LBUTTONDOWN:
        button = check_click((x, y), {rect:val[0] for rect, val in BUTTONS['MAIN'].items()})
        if button == 'SEGMENTOR':
            Segmentor(source).run()

def main(source:np.ndarray, path:str) -> bool:
    win_main, img_pos = draw_main_win(source)
    cv.namedWindow('MAIN')
    cv.setMouseCallback('MAIN', mouse_main, (source, win_main, img_pos))
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
    source_image, path = pick_source()
    repeat = main(source_image, path)
    while repeat:
        source_image, path = pick_source()
        repeat = main(source_image, path)
    print('Done.')