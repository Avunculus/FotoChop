import os
from chop import *

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
    global SEGMENTS
    SEGMENTS = []
    return 'sources/' + fn.split('.')[0] + '/'  # (image, 'sources/' + fn.split('.')[0] + '/')

def draw_main_win() -> np.ndarray:
    ...
    
def render() -> np.ndarray:
    ...


class Segment:
    def __init__(self, mask:np.ndarray):
        self.mask = mask # =None is flag for not visible/interacable
        self.render = True # whether to include it in the render
        self.source = 10  # 1-9: color canvas COLORS[i]; 10: src (color); 11: src (b&w)
        self.rect = (0, 0, 140, 60)
        self.rects = {'trash' : (  5, 195, 25, 30),
                      'render': ( 40, 195, 25, 30),
                      'source': ( 75, 195, 25, 30),
                      'up'    : (110, 190, 25, 20),
                      'down'  : (110, 210, 25, 20)}
    def delete(self):
        if self not in SEGMENTS: print(f'WARNING: attempt to remove unlisted segment')
        else: SEGMENTS.remove(self)
    def update_position(self) -> None:
        for rect in self.rects.values():
            pos = SEGMENTS.index(self)
            ... # move rect...
    def cycle_source(self, cycle_back=False) -> None:
        self.source += 1 if not cycle_back else -1
        self.source %= 11
    def toggle_render(self) -> None:
        self.render = not self.render
    def collide(self, pos:tuple[int,int]) -> bool:
        ... # collide check for whole rect: toggle highlighting
    def collide_rects(self, pos:tuple[int,int]) -> str|None:
        ...

def handle_mouse(event:int, x:int, y:int, flags:int, param):
    if event == cv.EVENT_LBUTTONDOWN:
        ... 
        # IF button 'chop' collide & len(SEGMENTS) < 9 ...
        # new_seg = Chopper(SOURCE).run()
        # if new_seg is not None: SEGMENTS.append(new_seg)
        # ELIF button render collide:
        # result = render()
        # show & save.


def main(path:str) -> bool:
    for i, seg in enumerate(os.listdir(path)):
        SEGMENTS.append(Segment(cv.imread(path + seg) // 255, i))
    # print(f'read {len(SEGMENTS)} segment masks from file.')
    cv.namedWindow('MAIN')
    cv.setMouseCallback('MAIN', handle_mouse)
    win_main = draw_main_win()
    while True:
        cv.imshow('MAIN', win_main)
        key = cv.waitKey(1)
        if key == 27:   break
        elif key == 18: # ctrl-r -> restart
            cv.destroyAllWindows()
            return True
        elif key > 0:   print(f'{key=}')
    cv.destroyAllWindows()
    return False

if __name__ == '__main__':
    path = pick_source()
    repeat = main(path)
    while repeat:
        path = pick_source()
        repeat = main(path)
    print('Done.')


#########################################################################################
#########################################################################################
#########################################################################################

# class Segment:
#     def __init__(self, mask:np.ndarray, position:int):
#         self.mask = mask # =None is flag for not visible/interacable
#         self.render = True # whether to include it in the render
#         self.source = 10  # 1-9: color canvas COLORS[i]; 10: src (color); 11: src (b&w)
#         self.pos = position
#         self.rect = (0, 180 + position * 60, 140, 60)
#     def cycle_source(self, cycle_back=False) -> None:
#         self.source += 1 if not cycle_back else -1
#         self.source %= 11
#     def toggle_render(self) -> None:
#         self.render = not self.render
#     def collide_check(self, pos:tuple[int,int]) -> str|bool:
#         if collision(self.rect, pos):
#             for name, rect in SEG_PANEL[self.pos]:
#                 if collision(rect, pos):
#                     return name
#             return True
#         return False



# def update_seg_order(segments:list[Segment]) -> None:
#     for i, seg in enumerate(segments):
#         seg.pos = i
#         seg.rect = (0, 180 + seg.pos * 60, 140, 60)




# def draw_main_win() -> np.ndarray:
#     win = square_frame(SOURCE, SIDE)
#     print(f'{win.shape[0] == win.shape[1]=}')
#     win = cv.copyMakeBorder(win, 0, 0, BAR, 0, cv.BORDER_CONSTANT, value=COLORS[0])
#     # callout btns: SEGMENTOR, RENDER
#     for (x, y, w, h), (name, color) in BUTTONS['MAIN'].items():
#         ix = np.ix_(np.arange(y, y + h), np.arange(x, x + w))
#         win[ix] = color
#         # put text....
#     # seg panels
#     for seg in SEGMENTS:
#         cv.rectangle(win, (0, 180 + seg.pos * PANEL_H), (BAR, 180 + PANEL_H + seg.pos * PANEL_H), COLORS[2], 6)
#         for name, rect in SEG_PANEL[seg.pos].items():
#             x, y, w, h = rect
#             match name:
#                 case 'trash'|'up'|'down':
#                     img = ASSETS[name]
#                 case 'render':
#                     img = ASSETS['on']
#                 case 'source':
#                     img = ASSETS['src color']
#             win[y: y + h, x: x + w, :] = img
#     return win
            


# def mouse_main(event:int, x:int, y:int, flags:int, param):
#     # source, win_main, img_pos = param
#     if event == cv.EVENT_LBUTTONDOWN:
#         for rect, (name, _) in BUTTONS['MAIN'].items():
#             if collision(rect, (x, y)):
#                 if name == 'SEGMENTOR':
#                     Segmentor().run()
#                 elif name == 'RENDER':
#                     ... # ! thread out for big time rendering!

#     elif event == cv.EVENT_MOUSEMOVE:
#         for seg in SEGMENTS:
    
#             if collision(seg.rect, (x, y)):
#                 ... # highlight segment
#                 print(f'{seg.pos}', end='|')


# def main(path:str) -> bool:
    
    
#     for i, seg in enumerate(os.listdir(path)):
#         SEGMENTS.append(Segment(cv.imread(path + seg) // 255, i))
    
#     print(f'read {len(SEGMENTS)} segment masks from file.')
#     if len(SEGMENTS) > 9:
#         print(f'warning! too many segment masks ({len(SEGMENTS)}) loaded.')
#         SEGMENTS = SEGMENTS[:9]
#     cv.namedWindow('MAIN')
#     cv.setMouseCallback('MAIN', mouse_main)
#     win_main = draw_main_win()
#     while True:
#         cv.imshow('MAIN', win_main)
#         key = cv.waitKey(1)
#         if key == 27:
#             break
#         elif key == 18:  # ctrl-r -> restart
#             cv.destroyAllWindows()
#             return True
#         elif key > 0:
#             print(f'{key=}')
#     cv.destroyAllWindows()
#     return False

# if __name__ == '__main__':
#     path = pick_source()
#     repeat = main(path)
#     while repeat:
#         path = pick_source()
#         repeat = main(path)
#     print('Done.')