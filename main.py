import os
from chop import *

def pick_source() -> str: #-> tuple[np.ndarray, str]:
    fn = 'uncle baby billy.jpg'
    ###
    name = 'sources/' + fn
    image = cv.imread(name)
    match image.shape[2]:   # set color format to 3-CHANNEL
        case 1: image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)  # grayscale
        case 4: image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)  # 4-channel
    global SOURCE
    SOURCE = image.copy()
    global PORTRAIT
    PORTRAIT = image.shape[0] > image.shape[1]
    global SEGMENTS
    SEGMENTS = []
    return 'sources/' + fn.split('.')[0] + '/'  # (image, 'sources/' + fn.split('.')[0] + '/')

def draw_buttons(win:np.ndarray, win_name:str) -> np.ndarray:
    for (x, y, w, h), (name, color) in BUTTONS[win_name].items():
        ix = np.ix_(np.arange(y, y + h), np.arange(x, x + w))
        win[ix] = color                                         # fill
        cv.rectangle(win, (x, y), (x + w, y + h), COLORS[2], 6) # outline
        cv.putText(win, name, (x + 6, y + h - 12), cv.FONT_HERSHEY_PLAIN, 1.5, 
                   COLORS[1], 2)  # text
    return win

def get_chop_window() -> np.ndarray:
    global SIZE
    win, SIZE = square_frame(SOURCE, SIDE)

    win = cv.copyMakeBorder(win, 0, 0, BAR, 0, cv.BORDER_CONSTANT, value=COLORS[0])
    win = draw_buttons(win, 'MAIN')
    for seg in SEGMENTS: seg.draw(win)
    return win
    
def render() -> np.ndarray:
    ...
#     def _save(self):
#         print('saving...', end='')
#         path = f'chopped/{self.job_name}.png'
#         # convert to 4-channel, set transparency
#         result = cv.cvtColor(self.result, cv.COLOR_BGR2BGRA) 
#         for i in range(result.shape[0]):    # laaaazyyyyy....
#             for j in range(result.shape[1]):
#                 result[i, j, 3] = 0 if not any(result[i, j, :3]) else 255
#         cv.imwrite(path, result, [cv.IMWRITE_PNG_COMPRESSION, 0])
#         cv.imwrite(f'chopped/{self.job_name}_MASK.jpg', self.bitmask)
#         print(f'...complete.\nSaved as: {path}')

class Segment:
    def __init__(self, mask:np.ndarray):
        self.mask = mask # =None is flag for not visible/interacable
        self.render = 1  # whether to include it in the render
        self.source = 9  # 0-8: color canvas COLORS[i]; 9: src (color); 10: src (b&w)
        self.highlight = False
    def delete(self):
        if self not in SEGMENTS: print(f'WARNING: attempt to remove unlisted segment')
        else: SEGMENTS.remove(self)
    def draw(self, win:np.ndarray) -> None:
        ix = SEGMENTS.index(self)
        up = int(ix != 0)
        down = int(ix + 1 < len(SEGMENTS))
        x, y, w, h = SEG_RECT
        y += ix * h
        win[np.ix_(np.arange(y, y + h), np.arange(x, x + w))] = COLORS[0] # fill/erase
        cv.rectangle(win, (x, y), (x + w, y + h), COLORS[2], 6)           # outline
        for name, (rect, images) in SEG_BTNS.items():                     # buttons
            x, y, w, h = rect
            y += ix * SEG_H
            match name:
                case 'trash':
                    img = images[0]
                case 'render':
                    img = images[self.render]
                case 'source':
                    img = images[self.source]
                case 'up':
                    img = images[up]
                case 'down':
                    img = images[down]
                case _: print('WARNING: unknown name from SEG_BTN')
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
            win[y: y + h, x: x + w, :] = img
    def cycle_source(self, cycle_back=False) -> None:
        self.source += 1 if not cycle_back else -1
        self.source %= 11
    def toggle_render(self) -> None:
        self.render = abs(self.render - 1)
    def move_up(self):
        ...
    def move_down(self):
        ...

def set_highlight(segment:Segment):
    frame, _ = square_frame(SOURCE, SIDE)
    gs_frame = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)
    gs_frame = np.stack([gs_frame, gs_frame, gs_frame], axis=2)
    mask, _ = square_frame(segment.mask, SIDE)
    gs_frame[np.nonzero(mask)] = frame[np.nonzero(mask)]
    WIN[:, BAR:, :] = gs_frame
    for seg in SEGMENTS: seg.highlight = False
    segment.highlight = True

def remove_highlight():
    frame, _ = square_frame(SOURCE, SIDE)
    WIN[:, BAR:, :] = frame
    for seg in SEGMENTS: seg.highlight = False

def draw_segments():
    WIN[180:, :BAR, :] *= 0
    for seg in SEGMENTS:
        seg.draw(WIN)

def handle_mouse(event:int, x:int, y:int, flags:int, param):
    if event == cv.EVENT_MOUSEMOVE:
        if x < BAR and y > 180:             # mouse in seg area
            seg_ix = (y - 180) // SEG_H
            if seg_ix < len(SEGMENTS):      # mouse in seg
                if not SEGMENTS[seg_ix].highlight:
                    set_highlight(SEGMENTS[seg_ix])
        elif any([seg.highlight for seg in SEGMENTS]):  # mouse exiting seg area
            remove_highlight()                          # set src_view -> gray
            
    elif event == cv.EVENT_LBUTTONDOWN:
        for ix, map in SEG_MAP.items():
            for name, rect in map.items():
                if collision(rect, (x, y)):
                    match name:
                        case 'trash':  SEGMENTS[ix].delete()
                        case 'render': SEGMENTS[ix].toggle_render()
                        case 'source': SEGMENTS[ix].cycle_source()
                        case 'up':     SEGMENTS[ix].move_up()
                        case 'down':   SEGMENTS[ix].move_down()
                    draw_segments()
        for rect, (name, _) in BUTTONS['MAIN'].items():
            if collision(rect, (x, y)): 
                if name == 'SEGMENT' and len(SEGMENTS) < 9:
                    # thread out!
                    new_seg = Chopper(SOURCE).run()
                    if new_seg is not None:
                        new_seg = Finisher(new_seg).run()
                        SEGMENTS.append(new_seg)
                elif name == 'RENDER':
                    # needs separate thread--will take a while
                    result = render()
                    ... # show (& save).


def main(path:str) -> bool:
    for seg in os.listdir(path):
        SEGMENTS.append(Segment(cv.imread(path + seg) // 255), )
    # print(f'read {len(SEGMENTS)} segment masks from file.')
    cv.namedWindow('MAIN')
    cv.setMouseCallback('MAIN', handle_mouse)
    global WIN
    WIN = get_chop_window()
    cv.imshow('MAIN', WIN)
    while True:
        cv.imshow('MAIN', WIN)
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


# class ImagePicker:
#     def __init__(self, sources:dict[str,np.ndarray]):
#         self.sources = sources      # {'xyz.png': thumbnail img}
#         self.box_h = 70
#         self.box_w = 70
#         txt_width = 500
#         self.w = self.box_w + txt_width
#         self.h = self.box_h * len(self.sources)
#         self.image = np.zeros((self.h, self.w, 3), np.uint8)
#         # draw thumbnails, filenames
#         x0, y0 = (3, 3)             # 3px buffer (each side) in x and y: 64x64 -> 70x70
#         for i, (fname, thumb) in enumerate(self.sources.items()):
#             h, w = thumb.shape[:2]
#             x = x0 + (self.box_w - w) // 2
#             y = (y0 + self.box_h * i) + (self.box_h - h) // 2
#             self.image[y : y + h, x : x + w] = thumb
#             cv.putText(self.image, fname, (76, y + self.box_h // 2),
#                        cv.FONT_HERSHEY_TRIPLEX, 1, [222, 222, 222])
#         self.y_offset = 0
#         self.max_offset = len(self.sources) % (H // self.box_h)
#         self.view = self.image[0 : min(self.h, H), :, :]
#         self.selections = list(self.sources.keys())
#         self.selected: str = None   # file name (WITH .ext)
#     def _update_view(self):
#         y0 = self.y_offset * self.box_h
#         self.view = self.image.copy()[y0 : y0 + self.view.shape[0], ...]
#         if self.selected:
#             point_a = (0, self.box_h * (self.selections.index(self.selected) - self.y_offset))
#             point_b = (self.w, point_a[1] + self.box_h)
#             cv.rectangle(self.view, point_a, point_b, BLUE, 3)
#     def _handle_mouse(self, event, x, y, *args):
#         if event == cv.EVENT_LBUTTONDOWN:
#             self.selected = self.selections[self.y_offset + y // self.box_h]
#             self._update_view()
#         elif event == cv.EVENT_MOUSEWHEEL:
#             self._scroll_view(pull_down=args[0] > 0)
#     def _scroll_view(self, pull_down:bool):
#         if pull_down and self.y_offset > 0: self.y_offset -= 1
#         elif not pull_down and self.y_offset < self.max_offset: self.y_offset += 1
#         self._update_view()
#     def run(self) -> tuple[np.ndarray,str]:
#         assert len(self.sources) > 0, 'no source images found'
#         cv.namedWindow('select source, then press spacebar')
#         cv.setMouseCallback('select source, then press spacebar', self._handle_mouse) 
#         running = True
#         while running:
#             cv.imshow('select source, then press spacebar', self.view)
#             key = cv.waitKey(1)
#             if key > 0: print(f'{key=}')
#             if key == 27: running = False       # esc: quit
#             elif key == 32 and self.selected:   # spc: accept
#                 cv.destroyAllWindows()
#                 return (cv.imread('source images/' + self.selected),
#                         self.selected.split('.')[0])
#         cv.destroyAllWindows()
#         return (None, '[user quit]')
