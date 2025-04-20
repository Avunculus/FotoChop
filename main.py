from threading import Thread
import os
from display import *

def pick_source() -> tuple[np.ndarray, str]:
    ###
    fn = 'uncle baby billy.jpg'
    ###
    name = 'sources/' + fn
    image = cv.imread(name)
    match image.shape[2]: # check Color format, convert to 3|4 channel?
        case 1: image = cv.cvtColor(image, cv.COLOR_GRAY2BGRA) # grayscale
        case 3: image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)  # 3-channel
    return (image, 'sources/' + fn.split('.')[0] + '/')

def manage_segments():
    ...

def collage():
    ...

def add_segment():
    ...

def click_main(event:int, x:int, y:int, flags:int, *args):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'left-clicked MAIN:\t{x=}\t{y=}\t{flags=}')


def main(source_image:np.ndarray, path:str) -> bool:
    # load existing segments
    if not os.access(path, mode=os.F_OK): # no dir exists
        os.mkdir(path)
    segments = []
    for seg in os.listdir(path):
        segments.append(cv.imread(path + seg))

    main_win = draw_main_win(source_image)
    cv.namedWindow('MAIN')
    cv.setMouseCallback('MAIN', click_main)
    while True:
        cv.imshow('MAIN', main_win)
        key = cv.waitKey(1)
        if key == 27: break            # esc: quit
    return False


if __name__ == '__main__':
    image, path = pick_source()
    repeat = main(image, path)
    while repeat:
        image, path = pick_source()
        repeat = main(image, path)
    print('Done.')
