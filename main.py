from threading import Thread, Event
import os
from queue import Queue
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
    return (image, 'sources/' + fn.split('.')[0] + '/')


def mouse_segment(event:int, x:int, y:int, flags:int, source:np.ndarray):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'clicked \'SEGMENT\' window @ ({x}, {y})')

def mouse_chopjob(event:int, x:int, y:int, flags:int, param):
    if event == cv.EVENT_LBUTTONDOWN:
        button = check_click((x, y), {rect:val[0] for rect, val in BUTTONS['CHOPJOB'].items()})
        if button:
            print(f'clicked on: {button}')


def mouse_main(event:int, x:int, y:int, flags:int, source:np.ndarray):
    if event == cv.EVENT_LBUTTONDOWN:
        button = check_click((x, y), {rect:val[0] for rect, val in BUTTONS['MAIN'].items()})
        if button == 'SEGMENTOR':
            # check if thread alive -> skip?
            # roi = (0, 0, 0, 0)  # get region of interest (x, y, w, h)
            # while any([i == 0 for i in roi[2:]]):
            img = scaledown_fit_view(cv.cvtColor(source, cv.COLOR_BGR2GRAY))
            roi = cv.selectROI('select ROI, then press spacebar', img)
            cv.destroyWindow('select ROI, then press spacebar')
            if all(roi[2:]):
                cv.namedWindow('SEGMENT')
                cv.imshow('SEGMENT', img)
                cv.setMouseCallback('SEGMENT', mouse_segment, source)
        elif button == 'CHOPJOB':
            cv.namedWindow('CHOPJOB')
            win_chopjob = draw_chopjob_win(source)
            cv.imshow('CHOPJOB', win_chopjob)
            cv.setMouseCallback('CHOPJOB', mouse_chopjob, win_chopjob)


def main(src_image:np.ndarray, src_dir:str) -> bool:
    global SEGMENTS
    SEGMENTS = Queue(9)
    # presaved segments to Q
    for seg in os.listdir(path):
        bitmask = cv.imread(path + seg) // 255
        SEGMENTS.put(bitmask)
    win_main = draw_main_win(src_image)
    cv.namedWindow('MAIN')
    cv.setMouseCallback('MAIN', mouse_main, src_image)
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