from threading import Thread
import os
from display import *

def pick_source() -> tuple[np.ndarray, str]:
    ###
    fn = 'ubb2.jpg'
    ###
    name = 'sources/' + fn
    image = cv.imread(name)
    # match image.shape[2]: # check Color format, convert to 3|4 channel?
    #     case 1: image = cv.cvtColor(image, cv.COLOR_GRAY2BGRA) # grayscale
    #     case 3: image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)  # 3-channel
    return (image, 'sources/' + fn.split('.')[0] + '/')

def manage_segments():
    ...

def collage():
    ...

def add_segment():
    ...



def main(source_image:np.ndarray, path:str) -> bool:
    # check for existing segments
    segs = os.listdir(path)

    source_view = resize_fit_view(source_image)
    cv.namedWindow('MAIN')
    while True:
        cv.imshow('MAIN', source_view)
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
