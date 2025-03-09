import numpy as np
import cv2 as cv
import os

W, H = (1920, 1080)
MAX_SIZE = 1000 * 1000
BLUE = [255,0,0]        # ROI
BLACK = [0,0,0]         # BG
WHITE = [255,255,255]   # FG
PURPLE = [255, 0, 255]  # reset to default ROI value = probable foreground
BRUSHES = {0: {'view': BLACK, 'mask': cv.GC_BGD},       # background
           1: {'view': WHITE, 'mask': cv.GC_FGD},       # foreground
           2: {'view': PURPLE, 'mask': cv.GC_PR_FGD}}   # edges/resets: pr_fgd

# 1 == cv.GC_FGD; 3==cv.GC_PR_FGD; 0,2 = BG,PRBG
# ! scaling: scale to fit VIEW_MAX (W, H) or SIZE_MAX (max grabcut size: set cap per timeout limit)
def scale_to_fit(img: np.ndarray, limit: int|tuple[int,int]) -> tuple[np.ndarray,int]:
    height, width = img.shape[:2]
    scale = 1
    if isinstance(limit, int):     too_big = lambda w, h, k: (w * h) // (k * k) > limit
    elif isinstance(limit, tuple): too_big = lambda w, h, k: w / k > limit[0] or h / k > limit[1]
    else: return
    while too_big(width, height, scale):
        scale += 1
    shape = (width // scale, height // scale)
    img = cv.resize(img, shape)
    return (img, scale)

def read_sources() -> dict[str,np.ndarray]:
    # creates thumbnails, namelist for available files (used by ImagePicker)
    thumbs = [n.removesuffix('.jpg') for n in os.listdir('src/thumbnails/')]
    fnames = [n for n in os.listdir('src/') if '.' in n]
    for fn in fnames: # 'xyz.jpg'
        name = fn[: fn.index('.')]
        if not name in thumbs: # make thumbnail
            img = cv.imread('src/' + fn)
            thumb, _ = scale_to_fit(img, (64, 64)) # cv.resize(img, (64, 64))
            cv.imwrite('src/thumbnails/' + name + '.jpg', thumb)
    sources = {}
    for fn in fnames: # 'xyz.jpg'
        name = fn[: fn.index('.')]
        sources[fn] = cv.imread('src/thumbnails/' + name + '.jpg')
    print(f'{len(sources)} available files')
    return sources

class ImagePicker:
    def __init__(self, sources:dict[str,np.ndarray]):
        self.sources = sources # {'xyz.png': thumbnail img}
        # make picker image: (thumb.w + txt.w, thumb.h * len(sources))
        self.box_h = 70
        self.box_w = 70
        txt_width = 500
        self.w = self.box_w + txt_width
        self.h = self.box_h * len(self.sources)
        self.image = np.zeros((self.h, self.w, 3), np.uint8) # np.ones((self.h, self.w, 3), np.uint8) * PURPLE
        # thumbs & text -> image
        x0, y0 = (3, 3)     # 3+3 buff in x and y: 64x64 -> 70x70
        for i, (fname, thumb) in enumerate(self.sources.items()):
            h, w = thumb.shape[:2]
            x = x0 + (self.box_w - w) // 2
            y = (y0 + self.box_h * i) + (self.box_h - h) // 2
            self.image[y : y + h, x : x + w] = thumb
            cv.putText(self.image, fname, (76, y + self.box_h // 2), cv.FONT_HERSHEY_TRIPLEX, 1, [222, 222, 222])
        self.y0 = 0                 # y origin. for scrolling, mapping clicks
        self.file_map = {i: key for i, key in enumerate(self.sources.keys())}
        self.view = self.image.copy()   # for draw refreshes
        self.selected: str = None   # file name [w/o ext]

    def _handle_mouse(self, event, x, y, *args):
        if event == cv.EVENT_LBUTTONDOWN:
            file_ix = (self.y0 + y) // self.box_h
            self.selected = self.file_map[file_ix]
            self.view = self.image.copy()
            cv.rectangle(self.view, (0, file_ix * self.box_h), (self.w, self.box_h + file_ix * self.box_h), BLUE, 3)
        elif event == cv.EVENT_MOUSEWHEEL:
            ... # FUTURE: scolling

    def run(self) -> tuple[np.ndarray,str]:
        if len(self.sources) == 0: return (None, '[no sources]') # if lib is empty post init, raise error & return
        cv.namedWindow('Select source image')
        cv.setMouseCallback('Select source image', self._handle_mouse) 
        running = True
        while running:
            cv.imshow('Select source image', self.view)
            key = cv.waitKey(1)
            if key == 27: running = False     # esc: quit
            elif key == 32 and self.selected:
                cv.destroyAllWindows()
                return (cv.imread('src/' + self.selected), self.selected.split('.')[0])   # spc: accept file
        cv.destroyAllWindows()
        return (None, None)

class GrabCutter:
    def __init__(self, image:np.ndarray, job_name:str):
        print(f'source image: \n{image.shape=}')
        self.source_full = image.copy()
        self.name = job_name
        # select roi, then set job attributes
        source, scale = scale_to_fit(image, (W, H))
        roi = (0, 0, 0, 0) # region of interest: (x, y, width, height)
        while all([i == 0 for i in roi]):
            roi = cv.selectROI(self.name + ' roi selection', source)
        cv.destroyWindow(self.name + ' roi selection')
        print(f'roi selected:\n{roi=} \n{scale=} ')
        x, y, w, h = [r * scale for r in roi]       # scale up roi rect
        roi = (x, y, w, h)
        # set source, re-scale
        self.source_roi = self.source_full[y: y + h, x: x + w]
        # print(f'full source roi: \n{self.source_full.shape=}\n{self.source_roi.shape=}')
        self.view_img, self.view_scale = scale_to_fit(self.source_roi, (W, H))
        # print(f'roi view rez scaled: \n{self.view_img.shape=}\n{self.view_scale=}')
        self.gc_img, self.gc_scale = scale_to_fit(self.source_roi, MAX_SIZE)
        # print(f'gc rez scaled: \n{self.gc_img.shape=}\n{self.gc_scale=}')
        self.gc_mask = np.ones(self.gc_img.shape[:2], dtype= np.uint8)
        self.gc_mask *= cv.GC_PR_FGD    # everything in ROI presumed probable FG
        self.bgm = np.zeros((1, 65), np.float64)    # background model, used by ._cut
        self.fgm = np.zeros((1, 65), np.float64)    # foreground model, used by ._cut
        self.cut_count: int = 0
        # caches
        self.view_img_cached = self.view_img.copy()     # for view refreshes & ._clear_draws
        self.gc_mask_cached = self.gc_mask.copy()       # for ._clear_draws
        # drawing & filtering
        self.drawing = -1       # 0 for drawing bg, 1 for drawing fg
        self.draw_rad = 3       # brush radius
        self.img_mask: np.ndarray = None    # bitmask for filtering image
        self.result: np.ndarray = None      # alpha channel image
        self.result_view: np.ndarray = None # scaled to (W, H) for display
        # self.temp_result = ...        # for undos

    def _cut(self):
        print('cutting...')
        self.gc_mask, self.bgm, self.fgm \
            = cv.grabCut(self.gc_img, self.gc_mask, None, self.bgm, self.fgm, 1, cv.GC_INIT_WITH_MASK)
        # make img_mask -> scale up -> make composite, set self.result= 
        mask = np.where((self.gc_mask==2)|(self.gc_mask==0), 0, 1).astype('uint8')
        h, w = self.source_roi.shape[:2]
        self.img_mask = cv.resize(mask, (w, h))
        result = self.source_roi * self.img_mask[:, :, np.newaxis]
        self.result = cv.cvtColor(result, cv.COLOR_BGR2BGRA)
        self.result_view, _ = scale_to_fit(self.result, (W, H))
        self.cut_count += 1
        self.view_img = self.view_img_cached.copy()
        self.gc_mask_cached = self.gc_mask.copy()  
        print(f'cut # {self.cut_count} complete:\tpx count = {np.count_nonzero(self.img_mask):,}')
        # debug
        mask = self.img_mask * 255
        cv.imwrite(f'bin/{self.name}_MASK__.png', mask)
        if self.cut_count == 0: cv.namedWindow('result')

    def _undo_cut(self):
        ...
        print('undoing cut')
        self.cut_count -= 1

    def _draw(self, x, y): # paralell draws to view, mask with scale conversion of x, y, draw radius
        cv.circle(self.view_img, (x, y), self.draw_rad, BRUSHES[self.drawing]['view'], -1)
        # scale for mask conversion: (x|y|w|h * view_scale) // gc_scale
        x0, y0, rad = [(n * self.view_scale) // self.gc_scale for n in (x, y, self.draw_rad)]
        cv.circle(self.gc_mask, (x0, y0), rad, BRUSHES[self.drawing]['mask'], -1)

    def _undo_draws(self):
        self.gc_mask = self.gc_mask_cached.copy()
        self.view_img = self.view_img_cached.copy()

    def _handle_mouse(self, event, x, y, *args):
        if event in [cv.EVENT_LBUTTONDOWN, cv.EVENT_RBUTTONDOWN]:
            self.drawing = int(event == cv.EVENT_LBUTTONDOWN)
            self._draw(x, y) 
        elif event == cv.EVENT_MOUSEMOVE and self.drawing >= 0:
            self._draw(x, y)
        elif event in [cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP]:
            self.drawing = -1

    def save(self):
        name = f'bin/{self.name}_CHOPPED__{self.cut_count} cuts__.png'
        cv.imwrite(name, self.result)

    def run(self):
        cv.namedWindow('ROI')
        cv.setMouseCallback('ROI', self._handle_mouse) 
        cv.moveWindow('ROI', 0, 0)     
        running = True
        while running:
            cv.imshow('ROI', self.view_img)
            if self.result is not None: cv.imshow('result', self.result_view)
            key = cv.waitKey(1)
            if   key == 27: running = False     # esc: quit
            elif key == 32: self._cut()         # spc: do grabcut
            # need switch for draw mode. adding 3rd color do draw over mistakes, etc.
            elif key == ord('s'): self.save()
            elif key == ord('c'): self._undo_draws()
            elif key == ord('z'): ...           # check for kmod ctrl -> self._undo_cut()
        cv.destroyAllWindows()

if __name__ == '__main__':
    print(__doc__)
    sources = read_sources()
    img, name = ImagePicker(sources).run()
    if img is not None: GrabCutter(img, name).run()
    print('Done')