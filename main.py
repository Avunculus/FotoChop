from threading import Thread, Event
import os
from queue import Queue
from display import *

SEGMENTS = Queue(9)

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

src, path = pick_source()
for seg in os.listdir(path):
    bitmask = cv.imread(path + seg) // 255
    SEGMENTS.put(bitmask)

def add_seg_mouse(event:int, x:int, y:int, flags:int, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'clicked ADD SEGMENT window @ ({x}, {y})')
    

def add_segment(source:np.ndarray, event:Event):
    event.wait()
    cv.namedWindow('ADD SEGMENT')
    # cv.setMouseCallback('ADD SEGMENT', add_seg_mouse, event)
    view = scaledown_fit_view(cv.cvtColor(source, cv.COLOR_BGR2GRAY))
    while True:
        cv.imshow('ADD SEGMENT', view)
        key = cv.waitKey(1)
        if key == 27:
            break
    

def main_mouse(event:int, x:int, y:int, flags:int, param:Event):
    if event == cv.EVENT_LBUTTONDOWN and x in range(10, 190) and y in range(10, 190):
        print(f'clicked BUTTON')
        param.set()

event = Event()
thread2 = Thread(target=add_segment, args=(src.copy(), event))
thread2.start()
src_view = draw_main_win(src)
cv.namedWindow('MAIN')
cv.setMouseCallback('MAIN', main_mouse, event)

while True:
    cv.imshow('MAIN', src_view)
    key = cv.waitKey(1)
    if key == 27:
        cv.destroyWindow('MAIN')
        thread2.join()
        break



# def add_segment(source_image:np.ndarray, segments:list[np.ndarray]):
#     ...

def click_main(event:int, x:int, y:int, flags:int, param:tuple[Thread,Thread]):
    segmentor, chopper = param
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'left-clicked MAIN:\t{x=}\t{y=}\t{flags=}')
        if not segmentor.is_alive():
            print('thread dead')
            # segmentor.start()



def main(source_image:np.ndarray, path:str) -> bool:
    # load existing segments
    if not os.access(path, mode=os.F_OK): # no dir exists
        os.mkdir(path)
    segments = []
    for seg in os.listdir(path):
        segments.append(cv.imread(path + seg))

    # declare subthreads
    segmentor = Thread(target=add_segment, args=(source_image, segments))
    # segmentor.start()
    chopper = Thread(target=print, args=(source_image, segments))
    main_win = draw_main_win(source_image)
    cv.namedWindow('MAIN')
    cv.setMouseCallback('MAIN', click_main, (segmentor, chopper))
    

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
