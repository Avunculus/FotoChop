from constants import *
PX_MAX  = 1000 * 1000
SIDE_CH = 1000
BAR_CH  = 240
BUF = 24
BTN_HEIGHT = 98
BUTTON_CH: dict[str,tuple[tuple,list]] = {}
for i, name in enumerate(['chop it!', 'undo cut', 'undo draw', 'finalize']):
    BUTTON_CH[name] = ((BUF, BUF + i * (BUF + BTN_HEIGHT), BAR_CH - (2 * BUF), BTN_HEIGHT), COLORS[i + 3])


BRUSHES = {0: {'view': COLORS[0], 'mask': cv.GC_BGD},       # background
           1: {'view': COLORS[1], 'mask': cv.GC_FGD}}       # foreground

def square_frame(image:np.ndarray, side:int, pad=True)-> tuple[np.ndarray, tuple]:
    """Returns image resized to fit long side into square frame, scaled image size as (w, h)
    If pad=True, returns with 0-padding on t&b|l&r (return image is a square)"""
    h, w  = image.shape[:2]
    size = (round(side * w / h), side) if h > w else (side, round(side * h / w))
    image = cv.resize(image, size)
    h, w  = image.shape[:2]
    if pad and abs(h - w) > 1:
        if image.shape[0] > image.shape[1]: # roi = portrait
            image = cv.copyMakeBorder(image, 0, 0, (side - w) // 2, (side - w) // 2, cv.BORDER_CONSTANT, value=COLORS[2]) #[0] * image.shape[2])
        else:
            image = cv.copyMakeBorder(image, (side - h) // 2, (side - h) // 2, 0, 0, cv.BORDER_CONSTANT, value=COLORS[2])
        # handle 1-px margin of error in //
        dx, dy = (abs(image.shape[1] - side), abs(image.shape[0] - side))
        if any([dx, dy]):
            image = cv.copyMakeBorder(image, dy, 0, dx, 0, cv.BORDER_CONSTANT, value=COLORS[2])
    return (image, size)

def integer_scaledown(image:np.ndarray, side_max=SIDE) -> tuple[np.ndarray,int]:
    scale = 1
    h, w = image.shape[:2]
    while h / scale > side_max or w / scale > side_max:
        scale += 1
    shape = (round(w / scale), round(h / scale))
    img = cv.resize(image, shape)
    return (img, scale)

def slice_rect(image:np.ndarray, rect:tuple) -> np.ndarray:
    x, y, w, h = rect
    return image[y: y + h, x: x + w, :]

def collision(rect:tuple[int,int,int,int], pos:tuple[int,int]) -> bool:
    x, y, w, h = rect
    if pos[0] in range(x, x + w) and pos[1] in range(y, y + h):
        return True
    return False

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

# def get_chop_window(source:np.ndarray)-> np.ndarray:
#         win, _ = square_frame(source, SIDE)
#         win = cv.copyMakeBorder(win, 0, 0, BAR, 0, cv.BORDER_CONSTANT, value=COLORS[3])
#         return win

def scale_max_pixels(source:np.ndarray, px_max) -> np.ndarray:
    h, w = source.shape[:2]
    while h * w > px_max:
        h *= .95
        w *= .95
    h, w = (round(h), round(w))
    print(f'resized from:\n{source.shape[:2]} [{source.shape[0] / source.shape[1]}] \nto: \n({h}, {w}) [{h / w}]')
    return cv.resize(source, (w, h))

def resize_rect(rect:tuple[int,int,int,int], image_src:np.ndarray,
                image_dest:np.ndarray) -> tuple[int,int,int,int]:
        fx = image_dest.shape[1] / image_src.shape[1]
        fy = image_dest.shape[0] / image_src.shape[0]
        x, y, w, h = rect
        return (round(n) for n in [x * fx, y * fy, w * fx, h * fy])

def draw_ch_buttons(win:np.ndarray) -> np.ndarray:
    for name, (rect, color) in BUTTON_CH.items():
        x, y, w, h = rect
        ix = np.ix_(np.arange(y, y + h), np.arange(x, x + w))
        win[ix] = color
        cv.rectangle(win, (x, y), (x + w, y + h), COLORS[2], 3)
        cv.putText(win, name, (x + 2, y + BTN_HEIGHT // 2),
                   cv.FONT_HERSHEY_COMPLEX, 1., COLORS[1], 2)
    return win

class Chopper:
    def __init__(self, source:np.ndarray):
        self.source = source
        win, size = square_frame(source, SIDE_CH)
        self.win  = cv.copyMakeBorder(win, 0, 0, BAR_CH, 0,
                                      cv.BORDER_CONSTANT, value=COLORS[7])
        self.win = draw_ch_buttons(self.win)
        w, h = size
        point = (BAR_CH + h - w, 0) if h > w else (BAR_CH, w - h) # (w, h)
        self.src_view  = cv.resize(self.source, size)
        self.view_rect = (point[0], point[1], w, h)
        self.view_pre_draw = self.src_view.copy()
        self.gc_source = scale_max_pixels(source, PX_MAX)
        h, w = self.gc_source.shape[:2]
        self.gc_mask = np.zeros((h, w))
        self.bgm = np.zeros((1, 65), np.float64)    # background model
        self.fgm = np.zeros((1, 65), np.float64)    # foreground model
        self.drawing   = -1       # 0=drawing BGD(?) (black), 1=FGD (white)
        self.draw_rad  = 3
        self.mask_final = None  # flag indicating cut has been finalized -> erode/dilate state

    def show_preview(self):
        bitmask = np.where((self.gc_mask==2)|(self.gc_mask==0), 0, 1).astype('uint8')
        preview = self.gc_source * bitmask[:, :, np.newaxis]
        preview = cv.resize(preview, (self.src_view.shape[1], self.src_view.shape[0]))
        rect = resize_rect(self.roi, self.gc_source, self.src_view)
        # crop out ROI: scale from gc_src -> src_view
        preview, _ = square_frame(slice_rect(preview, rect), SIDE, False)
        cv.imshow('PREVIEW', preview)
    def cut(self) -> None:
        self.mask_prev = self.gc_mask.copy()
        self.gc_mask, self.bgm, self.fgm = \
            cv.grabCut(self.gc_source, self.gc_mask, None,
                       self.bgm, self.fgm, 1, cv.GC_INIT_WITH_MASK)
        self.mask_bkup = self.gc_mask.copy()
        self.show_preview()  
    def undo_cut(self) -> None:
        self.gc_mask = self.mask_prev.copy()
        self.refresh_view()
        self.show_preview()
    def refresh_view(self) -> None:
        x, y = self.view_rect[:2]
        self.win[y: y + SIDE_CH, x: x + SIDE_CH, :] = self.src_view.copy()
        

    def draw(self, point:tuple[int,int]) -> None:
        # draw to win, draw to gc mask -> needs scale from src_view to gc_rez
        x, y = point
        x += self.view_rect[0] + BAR_CH
        y += self.view_rect[1]
        cv.circle(self.win, (x, y), self.draw_rad, BRUSHES[self.drawing]['view'], -1)
        ratio = self.gc_mask.shape[0] / self.src_view.shape[0]  # arbitrary x|y
        # ratio = max(ratio, 1 / ratio) # need??? (use resize_rect()?)
        x, y, rad = tuple([round(n * ratio) for n in [x, y, self.draw_rad]])
        cv.circle(self.gc_mask, (x , y), rad, BRUSHES[self.drawing]['mask'], -1)
        cv.imshow('CHOPPER', self.win)

    def undo_draw(self) -> None:
        self.gc_mask = self.mask_pre_draw.copy()
        self.refresh_view()
        # x, y, w, h = self.view_rect
        # self.win[y: y + h, x: x + w, :] = self.view_pre_draw
        cv.imshow('CHOPPER', self.win)

    def handle_mouse(self, event:int, x:int, y:int, flags:int, param):
            if collision(self.view_rect, (x, y)): # mouse in src_view
                # if event == cv.EVENT_LBUTTONDOWN:
                #     ...
                if event in [cv.EVENT_LBUTTONDOWN, cv.EVENT_RBUTTONDOWN]:
                    self.mask_pre_draw = self.gc_mask.copy()
                    x, y = self.view_rect[:2]
                    self.view_pre_draw = self.win[y: y + SIDE_CH, x: x + SIDE_CH, :].copy()
                    self.drawing = int(event==cv.EVENT_LBUTTONDOWN)
                    self.draw((x, y))
                elif event == cv.EVENT_MOUSEMOVE and self.drawing >= 0:
                    self.draw((x, y))
                elif event == cv.EVENT_LBUTTONUP|cv.EVENT_RBUTTONUP:
                    self.drawing = -1
            else:
                if event == cv.EVENT_MOUSEMOVE:
                    self.drawing = -1
                elif event == cv.EVENT_LBUTTONDOWN: # check buttons
                    for name, (rect, _) in BUTTON_CH.items():
                        if collision(rect, (x, y)):
                            match name:
                                case 'chop it!' : self.cut()
                                case 'undo cut' : self.undo_cut()
                                case 'undo draw': self.undo_draw()
                                case 'finalize' :
                                    cv.destroyWindow('PREVIEW')
                                    cv.destroyWindow('CHOPPER')
                                    bitmask = np.where((self.gc_mask==2)|(self.gc_mask==0), 0, 1).astype('uint8')
                                    self.mask_final = Finisher(bitmask).run()
                                    # return self.mask_final
                                    
    def run(self) -> np.ndarray|None:
        # get ROI: scaled to grabcut rez
        h, w = self.gc_source.shape[:2]
        self.roi = (0, 0, 0, 0)
        self.roi = cv.selectROI('select ROI, then press spacebar', self.gc_source)
        if not all(self.roi[2:]): self.roi = (1, 1, w - 1, h - 1)
        cv.destroyWindow('select ROI, then press spacebar')
        # first cut: init with rect
        self.gc_mask, self.bgm, self.fgm = \
            cv.grabCut(self.gc_source, self.gc_mask, self.roi, 
                       self.bgm, self.fgm, 1, cv.GC_INIT_WITH_RECT)
        self.mask_pre_draw = self.gc_mask.copy()
        cv.namedWindow('CHOPPER')
        cv.setMouseCallback('CHOPPER', self.handle_mouse)
        cv.imshow('CHOPPER', self.win)
        cv.namedWindow('PREVIEW')
        self.show_preview()





class Finisher:
    def __init__(self, mask:np.ndarray):
        self.mask = mask # copy?
    def erode(self):
        ...
    def dilate(self):
        ...
    def run(self) -> np.ndarray:
        return self.mask
