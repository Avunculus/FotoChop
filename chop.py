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

def get_window(image:np.ndarray) -> np.ndarray:
    win, size = square_frame(image, SIDE_CH)
    win  = cv.copyMakeBorder(win, 0, 0, BAR_CH, 0, cv.BORDER_CONSTANT, value=COLORS[7])
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
        self.win = get_window(source) # main win shows bitmask, pop-ups for draw & preview
        self.gc_source = scale_max_pixels(source, PX_MAX) # key for sizing src view window
        h, w = self.gc_source.shape[:2]
        self.src_view = cv.resize(source, (w, h))
        self.gc_mask = np.zeros((h, w)) # , dtype=np.uint8)   
        self.bgm = np.zeros((1, 65), np.float64)    # background model
        self.fgm = np.zeros((1, 65), np.float64)    # foreground model
        self.drawing  = -1       # 0=drawing BGD(?) (black), 1=FGD (white)
        self.draw_rad = 3
        self.mask_final = None  # flag indicating cut has been finalized -> erode/dilate state
        

    def show_preview(self):
        bitmask = np.where((self.gc_mask==2)|(self.gc_mask==0), 0, 1).astype('uint8')
        preview = self.gc_source * bitmask[:, :, np.newaxis]
        cv.namedWindow('PREVIEW')
        cv.imshow('PREVIEW', preview)
        bitmask = np.stack([bitmask, bitmask, bitmask], axis=2)
        self.win = get_window(bitmask * 255)
        cv.imshow('CHOPPER', self.win)

    def refresh_view(self) -> None:
        self.src_view = cv.resize(self.source, (self.gc_mask.shape[1], self.gc_mask.shape[0]))
        cv.namedWindow('SOURCE')
        cv.imshow('SOURCE', self.src_view) 

    def cut(self) -> None:
        self.mask_prev = self.gc_mask.copy()
        self.gc_mask, self.bgm, self.fgm = \
            cv.grabCut(self.gc_source, self.gc_mask, None,
                       self.bgm, self.fgm, 1, cv.GC_INIT_WITH_MASK)
        self.mask_bkup = self.gc_mask.copy()
        self.refresh_view()
        self.show_preview()

    def undo_cut(self) -> None:
        self.gc_mask = self.mask_prev.copy()
        self.refresh_view()
        self.show_preview()


    def draw(self, point:tuple[int,int]) -> None:
        x, y = point
        cv.circle(self.src_view, (x, y), self.draw_rad, BRUSHES[self.drawing]['view'], -1)
        cv.circle(self.gc_mask, (x , y), self.draw_rad, BRUSHES[self.drawing]['mask'], -1)
        cv.imshow('SOURCE', self.src_view) 

    def undo_draw(self) -> None:
        self.gc_mask = self.mask_pre_draw
        self.src_view = self.view_pre_draw
        cv.imshow('SOURCE', self.src_view)

    def mouse_draw(self, event:int, x:int, y:int, flags:int, param):
        if event in [cv.EVENT_LBUTTONDOWN, cv.EVENT_RBUTTONDOWN]:
            self.mask_pre_draw = self.gc_mask.copy()
            self.view_pre_draw = self.src_view.copy()
            self.drawing = int(event==cv.EVENT_LBUTTONDOWN)
            self.draw((x, y))
        elif event == cv.EVENT_MOUSEMOVE and self.drawing >= 0:
            self.draw((x, y))
        elif event in [cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP]:
            self.drawing = -1

    def mouse_main(self, event:int, x:int, y:int, flags:int, param):
        if event == cv.EVENT_LBUTTONDOWN: # check buttons
            for name, (rect, _) in BUTTON_CH.items():
                if collision(rect, (x, y)):
                    match name:
                        case 'chop it!' : self.cut()
                        case 'undo cut' : self.undo_cut()
                        case 'undo draw': self.undo_draw()
                        case 'finalize' :
                            self.mask_final = np.where((self.gc_mask==2)|(self.gc_mask==0), 0, 1).astype('uint8')
                            cv.destroyWindow('PREVIEW')
                            cv.destroyWindow('SOURCE')
                            cv.destroyWindow('CHOPPER')


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
        cv.namedWindow('SOURCE')
        cv.setMouseCallback('CHOPPER', self.mouse_main)
        cv.setMouseCallback('SOURCE', self.mouse_draw)
        cv.imshow('CHOPPER', self.win)
        self.show_preview()
        self.refresh_view()



K_SHAPE = [cv.MORPH_ERODE,
           cv.MORPH_DILATE,
           cv.MORPH_RECT,
           cv.MORPH_ELLIPSE]

class Finisher:
    def __init__(self, mask:np.ndarray):
        self.mask_original = mask
        self.mask = mask * 255
        self.mask_before = self.mask.copy()
        cv.namedWindow('FINALIZE', flags=cv.WINDOW_GUI_EXPANDED)
        cv.imshow('FINALIZE', self.mask)
        cv.createTrackbar('K SIZE', 'FINALIZE', 1, 12, self.trackbar_changed)
        cv.createTrackbar('K SHAPE', 'FINALIZE', 0, 3, self.trackbar_changed)
        
    def trackbar_changed(self, arg):
        ix = cv.getTrackbarPos('K SHAPE', 'FINALIZE')
        print(f'{arg=} | {ix=}')
        print(f'kernel size = {cv.getTrackbarPos('K SIZE', 'FINALIZE')} || kernel shape = {K_SHAPE[ix]}')

    def revert(self):
        self.mask = self.mask_original * 255   
        cv.imshow('FINALIZE', self.mask) 
    def morph(self, operation:str):
        size = cv.getTrackbarPos('K SIZE', 'FINALIZE')
        shape = K_SHAPE[cv.getTrackbarPos('K SHAPE', 'FINALIZE')]
        element = cv.getStructuringElement(shape, (2 * size + 1, 2 * size + 1), (size, size))
        self.mask_before = self.mask.copy()
        if operation == 'e':
            self.mask = cv.erode(self.mask, element)
        elif operation == 'd':
            self.mask = cv.dilate(self.mask, element)
        cv.imshow('FINALIZE', self.mask)
        
    # def run(self) -> np.ndarray:
    #     while True:
    #         cv.imshow('FINALIZE', self.mask)
    #         key = cv.waitKey(1)
    #         if   key == ord('e'): self.morph('e')
    #         elif key == ord('d'): self.morph('d')
    #         elif key == 32: # spc
    #             self.mask_final = self.mask
    #             break 
            # ctrl-z (30?) -> self.undo_op(): self.mask = self.mask_before

        
