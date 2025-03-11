import numpy as np
import cv2 as cv
import os

W, H = (1920, 1080)
MAX_SIZE = 1000 * 1000
BLUE = [255,0,0]        # ROI
BLACK = [0,0,0]         # BG
WHITE = [255,255,255]   # FG

BRUSHES = {0: {'view': BLACK, 'mask': cv.GC_BGD},       # background
           1: {'view': WHITE, 'mask': cv.GC_FGD}}       # foreground  # 2 == cv.GC_PR_BGD; 3 == cv.GC_PR_FGD

def scaledown_fit(img: np.ndarray, limit: int|tuple[int,int]) -> tuple[np.ndarray,int]:
    # integer scale image down to fit shape: (w, h) or size: pixels. Keeps aspect ratio.
    height, width = img.shape[:2]
    scale = 1
    if isinstance(limit, int):
        too_big = lambda w, h, k: (w * h) // (k * k) > limit
    elif isinstance(limit, tuple):
        too_big = lambda w, h, k: w / k > limit[0] or h / k > limit[1]
    else:
        return
    while too_big(width, height, scale):
        scale += 1
    shape = (width // scale, height // scale)
    img = cv.resize(img, shape)
    return (img, scale)

def read_sources() -> dict[str,np.ndarray]:
    # create thumbnails, namelist for available files (-> ImagePicker)
    thumbs = [n.removesuffix('.jpg') for n in os.listdir('src/thumbnails/')]
    fnames = os.listdir('src/')
    for fn in fnames:               # 'xyz.jpg'
        name = fn[: fn.split('.')[0]]
        if name not in thumbs:      # make thumbnail
            img = cv.imread('src/' + fn)
            thumb, _ = scaledown_fit(img, (64, 64)) # cv.resize(img, (64, 64))
            cv.imwrite('src/thumbnails/' + name + '.jpg', thumb)
    sources = {}
    for fn in fnames:
        name = fn[: fn.index('.')]
        sources[fn] = cv.imread('src/thumbnails/' + name + '.jpg')
    return sources

class ImagePicker:
    def __init__(self, sources:dict[str,np.ndarray]):
        self.sources = sources      # {'xyz.png': thumbnail img}
        self.box_h = 70
        self.box_w = 70
        txt_width = 500
        self.w = self.box_w + txt_width
        self.h = self.box_h * len(self.sources) # min(H, box_h * len(sources))... add scrolling?
        self.image = np.zeros((self.h, self.w, 3), np.uint8)
        x0, y0 = (3, 3)                 # 3+3 buff in x and y: 64x64 -> 70x70
        for i, (fname, thumb) in enumerate(self.sources.items()):
            h, w = thumb.shape[:2]
            x = x0 + (self.box_w - w) // 2
            y = (y0 + self.box_h * i) + (self.box_h - h) // 2
            self.image[y : y + h, x : x + w] = thumb
            cv.putText(self.image, fname, (76, y + self.box_h // 2), cv.FONT_HERSHEY_TRIPLEX, 1, [222, 222, 222])
        self.y0 = 0                     # y origin. for scrolling / mapping clicks
        self.file_map = {i: key for i, key in enumerate(self.sources.keys())}
        self.view = self.image.copy()   # for draw refreshes
        self.selected: str = None       # file name [w/o ext]

    def _handle_mouse(self, event, x, y, *args):
        if event == cv.EVENT_LBUTTONDOWN:
            file_ix = (self.y0 + y) // self.box_h
            self.selected = self.file_map[file_ix]
            self.view = self.image.copy()
            cv.rectangle(self.view, (0, file_ix * self.box_h), (self.w, self.box_h + file_ix * self.box_h), BLUE, 3)
        elif event == cv.EVENT_MOUSEWHEEL: ... # FUTURE: scolling

    def run(self) -> tuple[np.ndarray,str]:
        if len(self.sources) == 0:      # if empty post init, raise error & return
            return (None, '[no sources]') 
        cv.namedWindow('Select source image')
        cv.setMouseCallback('Select source image', self._handle_mouse) 
        running = True
        while running:
            cv.imshow('Select source image', self.view)
            key = cv.waitKey(1)
            if key == 27: running = False       # esc: quit
            elif key == 32 and self.selected:   # spc: accept
                cv.destroyAllWindows()
                return (cv.imread('src/' + self.selected), self.selected.split('.')[0])
        cv.destroyAllWindows()
        return (None, None)

class GrabCutter:
    def __init__(self, image:np.ndarray, job_name:str):
        print(f'source image: \n{image.shape=}')
        self.source_full = image.copy()
        self.name = job_name
        # select roi
        source, scale = scaledown_fit(image, (W, H))
        roi = (0, 0, 0, 0) # region of interest: (x, y, width, height)
        while all([i == 0 for i in roi]):
            roi = cv.selectROI(self.name + ' roi selection', source)
        cv.destroyWindow(self.name + ' roi selection')
        x, y, w, h = [r * scale for r in roi]       # scale up roi rect
        roi = (x, y, w, h)
        # re-scale, set params
        self.source_roi = self.source_full[y: y + h, x: x + w]
        self.view_img, self.view_scale = scaledown_fit(self.source_roi, (W, H))
        self.gc_img, self.gc_scale = scaledown_fit(self.source_roi, MAX_SIZE)
        self.gc_mask = np.ones(self.gc_img.shape[:2], dtype= np.uint8)
        self.gc_mask *= cv.GC_PR_FGD    # everything in ROI presumed probable FG
        self.bgm = np.zeros((1, 65), np.float64)    # background model
        self.fgm = np.zeros((1, 65), np.float64)    # foreground model
        self.cut_count: int = 0
        # caches
        self.view_img_cached = self.view_img.copy()     # for view refreshes & ._clear_draws()
        self.gc_mask_cached = self.gc_mask.copy()       # for ._clear_draws()
        # drawing & filtering
        self.drawing = -1       # 0 for drawing bg, 1 for drawing fg
        self.draw_rad = 3       # brush radius
        self.img_mask: np.ndarray = None    # bitmask for filtering image
        self.result: np.ndarray = None      # alpha channel image
        self.result_view: np.ndarray = None # scaled to (W, H) for display
        self.temp_result = np.ndarray = None        # for undos

    def _cut(self):
        print('cutting...')
        self.gc_mask, self.bgm, self.fgm \
            = cv.grabCut(self.gc_img, self.gc_mask, None, self.bgm, self.fgm, 1, cv.GC_INIT_WITH_MASK)
        # make img_mask -> scale up -> make composite, set self.result= 
        mask = np.where((self.gc_mask==2)|(self.gc_mask==0), 0, 1).astype('uint8')
        h, w = self.source_roi.shape[:2]
        self.img_mask = cv.resize(mask, (w, h))
        result = self.source_roi * self.img_mask[:, :, np.newaxis]
        result = cv.cvtColor(result, cv.COLOR_BGR2BGRA) # ?
        # cache prev result
        # messed this up... don;'t need to keep resultant image cpoy for undos--JUST MASK. re-do attributes...
        self.temp_result = self.result.copy() if self.result else result
        self.result = result
        self.result_view, _ = scaledown_fit(self.result, (W, H))
        self.cut_count += 1
        self.view_img = self.view_img_cached.copy()
        self.gc_mask_cached = self.gc_mask.copy()  
        print(f'cut # {self.cut_count} complete:\tpx count = {np.count_nonzero(self.img_mask):,}')
        # !! make mask window viewable for feeedback
        mask = self.img_mask * 255
        cv.imwrite(f'bin/{self.name}_MASK__.png', mask)
        if self.cut_count == 0: cv.namedWindow('result')

    def _undo_cut(self):
        ... # reset mask back 2 steps; re-draw composite 
        print('undoing cut')
        self.cut_count -= 1

    def _draw(self, x, y):      # paralell draws to view, mask with scale conversion of x, y, draw radius
        cv.circle(self.view_img, (x, y), self.draw_rad, BRUSHES[self.drawing]['view'], -1)
        # scale for mask conversion: (x|y|w|h * view_scale) // gc_scale
        x0, y0, rad = [(n * self.view_scale) // self.gc_scale for n in (x, y, self.draw_rad)]
        cv.circle(self.gc_mask, (x0, y0), rad, BRUSHES[self.drawing]['mask'], -1)

    def _undo_draws(self):
        self.gc_mask = self.gc_mask_cached.copy()
        self.view_img = self.view_img_cached.copy()

    def _handle_mouse(self, event, x, y, flags, *args):
        if event in [cv.EVENT_LBUTTONDOWN, cv.EVENT_RBUTTONDOWN]:
            self.drawing = int(event == cv.EVENT_LBUTTONDOWN)
            self._draw(x, y) 
        elif event == cv.EVENT_MOUSEMOVE and self.drawing >= 0:
            self._draw(x, y)
        elif event in [cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP]:
            self.drawing = -1

    def _save(self):
        name = f'bin/{self.name}_CHOPPED__{self.cut_count} cuts__.png'
        cv.imwrite(name, self.result)

    def run(self) -> bool:
        cv.namedWindow('ROI')
        cv.setMouseCallback('ROI', self._handle_mouse) 
        cv.moveWindow('ROI', 0, 0)     
        running = True
        while running:
            cv.imshow('ROI', self.view_img)
            if self.result is not None:
                cv.imshow('result', self.result_view)
            key = cv.waitKey(1)
            if   key == 27: running = False     # esc: quit
            elif key == 32: self._cut()         # spc: do grabcut
            elif key == ord('s'): self._save()
            elif key == ord('c'): self._undo_draws()
            elif key == ord('r'):               # restart
                cv.destroyAllWindows()
                return True
            elif key == ord('z'): # and cv.EVENT_FLAG_CTRLKEY: # how to check for kmod ctrl??? -> self._undo_cut()
                print('z')
            elif key == ord('z') & cv.EVENT_FLAG_CTRLKEY:
                print('ctrl-z')
        cv.destroyAllWindows()
        return False

if __name__ == '__main__':
    sources = read_sources()
    img, name = ImagePicker(sources).run()
    repeat = GrabCutter(img, name).run() if img is not None else False
    while repeat:
        img, name = ImagePicker(sources).run()
        repeat = GrabCutter(img, name).run() if img is not None else False   
    print('Done')