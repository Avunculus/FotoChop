import os
from chop import *
from system import ImagePicker, save_render, write_segments

def pick_source() -> str: #-> tuple[np.ndarray, str]:
    fn = ImagePicker().run()
    cv.destroyWindow('PICK IMAGE')
    if fn == '':
        print('Failed to pick filename')
        return ''
    name = 'sources/' + fn
    image = cv.imread(name)
    match image.shape[2]:   # set color format to 3-CHANNEL
        case 1: image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)  # grayscale
        case 4: image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)  # 4-channel
    global SOURCE
    SOURCE = image.copy()
    global PORTRAIT
    PORTRAIT = image.shape[0] > image.shape[1]
    global BACKGROUND
    BACKGROUND = BackGround(np.ones(image.shape[:2]))
    global SEGMENTS
    SEGMENTS = [BACKGROUND, ]
    return 'sources/' + fn.split('.')[0] + '/'  # (image, 'sources/' + fn.split('.')[0] + '/')

def draw_buttons(win:np.ndarray, win_name:str) -> np.ndarray:
    for (x, y, w, h), (name, color) in BUTTONS[win_name].items():
        ix = np.ix_(np.arange(y, y + h), np.arange(x, x + w))
        win[ix] = color                                         # fill
        cv.rectangle(win, (x, y), (x + w, y + h), COLORS[2], 3) # outline
        cv.putText(win, name, (x + 6, y + h - 12),
                   cv.FONT_HERSHEY_PLAIN, 1., COLORS[1], 2)  # text
    return win

def get_window() -> np.ndarray:
    global SIZE
    win, SIZE = square_frame(SOURCE, SIDE)
    win = cv.copyMakeBorder(win, 0, 0, BAR, 0, cv.BORDER_CONSTANT, value=COLORS[0])
    win = draw_buttons(win, 'MAIN')
    for seg in SEGMENTS: seg.draw(win)
    return win

def render() -> np.ndarray:
    result = cv.cvtColor(SOURCE, cv.COLOR_BGR2BGRA) * 0
    
    for seg in SEGMENTS:
        if seg.render:
            mask = cv.resize(seg.mask, (result.shape[1], result.shape[0]))
            if seg.source < 9: # solid color
                src = np.ones_like(result) * COLORS[seg.source]
                src[..., 3] = 255
            else: 
                match seg.source: # 0-8 = colors; 9, 10, 11 = img, imggray, alpha0
                    case 9: src = cv.cvtColor(SOURCE, cv.COLOR_BGR2BGRA)
                    case 10:
                        src = cv.cvtColor(SOURCE, cv.COLOR_BGR2GRAY)
                        src = np.stack([src, src, src, np.ones(src.shape[:2]) * 255], axis=2)
                    case 11: src = np.zeros_like(result)
            result[np.nonzero(mask)] = src[np.nonzero(mask)]
    return result



class Segment:
    def __init__(self, mask:np.ndarray):
        self.mask = mask # size = grabcut rez(?)
        self.render = 1  # whether to include it in the render
        self.source = 9  # 0-8: color canvas COLORS[i]; 9: src (color); 10: src (b&w); 11: alpha-transp [0, 0, 0, 0]
        self.highlight = False
    def delete(self):
        if self not in SEGMENTS: print(f'WARNING: attempt to remove unlisted segment')
        else: SEGMENTS.remove(self)
    def draw(self, win:np.ndarray) -> None:
        ix = SEGMENTS.index(self)
        up = int(ix > 1)  # BG always ix0
        down = int(ix + 1 < len(SEGMENTS))
        x, y, w, h = SEG_RECT
        y += ix * h
        # win[np.ix_(np.arange(y, y + h), np.arange(x, x + w))] = COLORS[0] # fill/erase
        cv.rectangle(win, (x, y), (x + w, y + h), COLORS[7], 3)           # outline
        # # SEG_MAP method:
        # for name, rect in SEG_MAP[ix].items():
        # direct method:
        for name, (rect, images) in SEG_BTNS.items():                     # segs
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
        self.source %= 12
    def toggle_render(self) -> None:
        self.render = abs(self.render - 1)
    def move_up(self):
        ix = SEGMENTS.index(self)
        if ix > 1:
            SEGMENTS.remove(self)
            SEGMENTS.insert(ix - 1, self)
    def move_down(self):
        ix = SEGMENTS.index(self)
        if ix < len(SEGMENTS) - 1:
            SEGMENTS.remove(self)
            SEGMENTS.insert(ix + 1, self)

class BackGround(Segment):
    def __init__(self, mask):
        super().__init__(mask)
    def delete(self):
        pass
    def draw(self, win:np.ndarray) -> None:
        # win[np.ix_(np.arange(y, y + h), np.arange(x, x + w))] = COLORS[0] # fill/erase
        # cv.rectangle(win, (x, y), (x + w, y + h), COLORS[7], 3)           # outline
        for name, (rect, images) in SEG_BTNS.items():   # seg buttons
            x, y, w, h = rect
            if name == 'render':
                win[y: y + h, x: x + w, :] = images[self.render]
            elif name == 'source':
                img = images[self.source] 
                if img.ndim == 2: img = np.stack([img, img, img], axis=2)
                win[y: y + h, x: x + w, :] = img
    def move_down(self):
        pass
    def move_up(self):
        pass


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
    WIN[120:, :BAR, :] *= 0
    for seg in SEGMENTS:
        seg.draw(WIN)

def handle_mouse(event:int, x:int, y:int, flags:int, param):
    if event == cv.EVENT_MOUSEMOVE:
        if x < BAR and y > 120:             # mouse in seg area
            seg_ix = (y - 120) // SEG_H
            if seg_ix < len(SEGMENTS) and seg_ix != 0:      # mouse in seg
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
                match name:
                    case 'SEGMENT':
                        if len(SEGMENTS) < 10:
                            global CHOPPER
                            CHOPPER = Chopper(SOURCE)
                            CHOPPER.run()
                    case 'TAKE MASK':
                        if CHOPPER.mask_final is not None and len(SEGMENTS) < 10:
                            cv.namedWindow('FINALIZE')
                            global FINISHER
                            FINISHER = Finisher(CHOPPER.mask_final)
                            # FINISHER.run()
                    # case 'FINALIZE':
                    #     if FINISHER.mask_final is not None:
                    #         cv.destroyWindow('FINALIZE')
                    #         SEGMENTS.append(Segment(FINISHER.mask_final))
                    #         draw_segments()


def main(path:str) -> bool:
    if not os.path.isdir(path):
        os.mkdir(path)
    for seg in [n for n in os.listdir(path) if '.' in n]:
        SEGMENTS.append(Segment(cv.imread(path + seg) // 255), )
    # print(f'read {len(SEGMENTS)} segment masks from file.')
    global FINISHER
    FINISHER = None
    cv.namedWindow('MAIN')
    cv.setMouseCallback('MAIN', handle_mouse)
    global WIN
    WIN = get_window()
    cv.imshow('MAIN', WIN)
    while True:
        cv.imshow('MAIN', WIN)
        key = cv.waitKey(1)
        if key == 27:   break
        elif key == 23:  # ctrl-w
            write_segments(path, [s.mask * 255 for s in SEGMENTS[1:]])
        elif key == 18: # ctrl-r -> restart
            cv.destroyAllWindows()
            return True
        elif key == ord('e') and FINISHER is not None: FINISHER.morph('e')
        elif key == ord('d') and FINISHER is not None: FINISHER.morph('d')
        elif key == 32 and FINISHER is not None:
            cv.destroyWindow('FINALIZE')
            SEGMENTS.append(Segment(FINISHER.mask // 255))
            draw_segments()
        elif key == 13: # enter
            result = render()
            name = save_render(result, path)
            cv.namedWindow(f'RENDER: {name}', flags=cv.WINDOW_KEEPRATIO) # WINDOW_KEEPRATIO # ?
            cv.imshow(f'RENDER: {name}', result)
            ... # show
        elif key > 0:   print(f'{key=}')
    cv.destroyAllWindows()
    return False

if __name__ == '__main__':
    path = pick_source()
    repeat = False
    if path != '':
        repeat = main(path)
    while repeat:
        path = pick_source()
        if path == '': break
        repeat = main(path)
    print('Done.')
