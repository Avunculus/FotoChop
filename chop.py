import numpy as np
import cv2 as cv
import os

W, H = (1920, 1080)
GC_MAX_PIXELS = 1000 * 1000  # pixel cap for scaling images sent to cv.grabcut()
BLUE = [255,0,0]
BLACK = [0,0,0]
WHITE = [255,255,255]

BRUSHES = {0: {'view': BLACK, 'mask': cv.GC_BGD},       # background
           1: {'view': WHITE, 'mask': cv.GC_FGD}}       # foreground
# 2 == cv.GC_PR_BGD; 3 == cv.GC_PR_FGD

def scaledown_fit(img: np.ndarray, limit: int|tuple[int,int]) -> tuple[np.ndarray,int]:
    # integer scale image down to fit shape: (w, h) or size: pixels. Keeps aspect ratio.
    assert isinstance(limit, tuple) or isinstance(limit, int), f'bad limit dtype: {limit}'
    h, w = img.shape[:2]
    scale = 1
    while (isinstance(limit, int) and (w * h) // (scale * scale) > limit) or \
        (isinstance(limit, tuple) and (w / scale > limit[0] or h / scale > limit[1])):
        scale += 1
    shape = (w // scale, h // scale)
    img = cv.resize(img, shape)
    return (img, scale)
    # OLD:
    # if   isinstance(limit, int)  : too_big = lambda w, h, k: (w * h) // (k * k) > limit
    # elif isinstance(limit, tuple): too_big = lambda w, h, k: w / k > limit[0] or h / k > limit[1]
    # else: return
    # while too_big(width, height, scale):
    #     scale += 1
    # shape = (width // scale, height // scale)
    
def read_sources() -> dict[str,np.ndarray]:
    """Returns {filename: thumbnail image} for all images in 'source images/' """
    thumbs = [n.removesuffix('.jpg') for n in os.listdir('source images/thumbnails/')]
    fnames = [n for n in os.listdir('source images/') if '.' in n]
    # create thumbnails
    for fn in fnames:           # 'xyz.jpg'
        name = fn.split('.')[0]
        if name not in thumbs:
            img = cv.imread('source images/' + fn)
            thumb, _ = scaledown_fit(img, (64, 64))
            cv.imwrite('source images/thumbnails/' + name + '.jpg', thumb)
    sources = {}
    for fn in fnames:
        name = fn.split('.')[0]
        sources[fn] = cv.imread('source images/thumbnails/' + name + '.jpg')
    return sources


class ImagePicker:
    def __init__(self, sources:dict[str,np.ndarray]):
        self.sources = sources      # {'xyz.png': thumbnail img}
        self.box_h = 70
        self.box_w = 70
        txt_width = 500
        self.w = self.box_w + txt_width
        self.h = self.box_h * len(self.sources)     # FUTURE: need upper bound or scrolling... min(H, self.box_h * len(sources))?
        self.image = np.zeros((self.h, self.w, 3), np.uint8)
        x0, y0 = (3, 3)             # 3px buffer (each side) in x and y: 64x64 -> 70x70
        for i, (fname, thumb) in enumerate(self.sources.items()):
            h, w = thumb.shape[:2]
            x = x0 + (self.box_w - w) // 2
            y = (y0 + self.box_h * i) + (self.box_h - h) // 2
            self.image[y : y + h, x : x + w] = thumb
            cv.putText(self.image, fname, (76, y + self.box_h // 2),
                       cv.FONT_HERSHEY_TRIPLEX, 1, [222, 222, 222])
        self.y0 = 0                 # y origin; for scrolling / mapping clicks
        self.file_map = {i: key for i, key in enumerate(self.sources.keys())}
        self.view = self.image.copy()
        self.selected: str = None   # file name (WITH .ext)

    def _handle_mouse(self, event, x, y, *args):
        if event == cv.EVENT_LBUTTONDOWN:
            file_ix = (self.y0 + y) // self.box_h
            self.selected = self.file_map[file_ix]
            self.view = self.image.copy()       # copy source image to clear previous rect
            cv.rectangle(self.view, (0, file_ix * self.box_h),
                         (self.w, self.box_h + file_ix * self.box_h), BLUE, 3)
        elif event == cv.EVENT_MOUSEWHEEL: ...  # FUTURE: scolling

    def run(self) -> tuple[np.ndarray,str]:
        assert len(self.sources) > 0, 'no source images found'
        cv.namedWindow('select source, then press spacebar')
        cv.setMouseCallback('select source, then press spacebar', self._handle_mouse) 
        running = True
        while running:
            cv.imshow('select source, then press spacebar', self.view)
            key = cv.waitKey(1)
            if key == 27: running = False       # esc: quit
            elif key == 32 and self.selected:   # spc: accept
                cv.destroyAllWindows()
                return (cv.imread('source images/' + self.selected), self.selected.split('.')[0])
        cv.destroyAllWindows()
        return (None, '[user quit]')


class GrabCutter:
    def __init__(self, image:np.ndarray, job_name:str):
        print(f'SOURCE:\n{job_name=}\n{image.shape=}')
        # image coming in as 3-channel BGR
        self.name = job_name
        # select ROI (region of interest)
        source, scale = scaledown_fit(image, (W, H))
        roi = (0, 0, 0, 0)      # (x, y, width, height)
        while any([i == 0 for i in roi[2:]]):
            roi = cv.selectROI('select ROI, then press spacebar', source)
        cv.destroyWindow('select ROI, then press spacebar')
        x, y, w, h = [r * scale for r in roi]   # scale back up to index ROI
        roi = (x, y, w, h)
        # source, result, views
        self.source = image[y: y + h, x: x + w] # source = ROI; outside roi forgotten
        self.source_view, self.view_scale = scaledown_fit(self.source.copy(), (W, H))
        self.result = np.zeros_like(self.source)
        self.result_view = self.source_view.copy()
        # cv.grabcut() args
        self.gc_source, self.gc_scale = scaledown_fit(self.source, GC_MAX_PIXELS)
        self.gc_mask = np.ones(self.gc_source.shape[:2], dtype= np.uint8) * cv.GC_PR_FGD    # all px presumed probable FG
        self.bgm = np.zeros((1, 65), np.float64)    # background model
        self.fgm = np.zeros((1, 65), np.float64)    # foreground model
        self.cut_count = 0
        # draws and undos
        self.drawing  = -1       # 0 = drawing bg, 1 = drawing fg
        self.draw_rad =  3       # brush radius
        self.pre_cut  = (self.gc_mask.copy(), self.bgm.copy(), self.fgm.copy())    # for undoing cuts
        self.mask_post_cut = self.gc_mask.copy()    # for clearing draws

    def _update_result(self):
        # updates result img & view: convert gc_mask to 0|1
        mask = np.where((self.gc_mask==2)|(self.gc_mask==0), 0, 1).astype('uint8')
        # scale up, apply to source -> set self.result= -> scale down to viewsize
        h, w = self.source.shape[:2]
        mask = cv.resize(mask, (w, h))
        self.result = self.source * mask[:, :, np.newaxis]
        self.result_view = cv.resize(self.result, (w * self.view_scale, h * self.view_scale))
        # save b/w mask
        mask *= 255
        cv.imwrite(f'chopped/{self.name}_MASK_.jpg', mask)
        # reset view for new draws
        self.source_view = self.source.copy() 

    def _cut(self):
        print('cutting...')
        self.pre_cut = (self.gc_mask.copy(), self.bgm.copy(), self.fgm.copy()) 
        self.gc_mask, self.bgm, self.fgm = \
            cv.grabCut(self.gc_source, self.gc_mask, None, self.bgm, self.fgm, 1, cv.GC_INIT_WITH_MASK)
        self.mask_post_cut = self.gc_mask.copy()
        self.cut_count += 1
        print(f'cut # {self.cut_count} complete')
        self._update_result()
        # OLD:
        # print('cutting...')
        # self.gc_mask, self.bgm, self.fgm \
        #     = cv.grabCut(self.gc_img, self.gc_mask, None, self.bgm, self.fgm, 1, cv.GC_INIT_WITH_MASK)
        # make img_mask -> scale up -> make composite, set self.result= 
        # mask = np.where((self.gc_mask==2)|(self.gc_mask==0), 0, 1).astype('uint8')
        # h, w = self.source_roi.shape[:2]
        # self.img_mask = cv.resize(mask, (w, h))
        # result = self.source_roi * self.img_mask[:, :, np.newaxis]
        # result = cv.cvtColor(result, cv.COLOR_BGR2BGRA) # ?
        # cache prev result:
        # messed this up... don;'t need to keep resultant image cpoy for undos--JUST MASK. re-do attributes...
        # self.temp_result = self.result.copy() if self.result else result
        # self.result = result
        # self.result_view, _ = scaledown_fit(self.result, (W, H))
        # self.cut_count += 1
        # self.view_img = self.view_img_cached.copy()
        # self.gc_mask_cached = self.gc_mask.copy()  
        # print(f'cut # {self.cut_count} complete:\tpx count = {np.count_nonzero(self.img_mask):,}')
        # # !! make mask window viewable for feeedback
        # mask = self.img_mask * 255
        # cv.imwrite(f'bin/{self.name}_MASK__.png', mask)
        

    def _undo_cut(self):
        if self.cut_count < 1:
            return
        print('undoing cut')
        self.gc_mask, self.bgm, self.fgm = [m.copy() for m in self.pre_cut]
        self.cut_count -= 1
        self._update_result()

    def _draw(self, x, y):
        # draw to view, scaled up draw to mask: (x|y|w|h * view_scale) // gc_scale
        cv.circle(self.source_view, (x, y), self.draw_rad, BRUSHES[self.drawing]['view'], -1)
        x0, y0, rad = [(n * self.view_scale) // self.gc_scale for n in (x, y, self.draw_rad)]
        cv.circle(self.gc_mask, (x0, y0), rad, BRUSHES[self.drawing]['mask'], -1)

    def _clear_draws(self):
        self.gc_mask = self.mask_post_cut.copy()
        self.source_view = self.source.copy()

    def _handle_mouse(self, event, x, y, flags, *args):
        if event in [cv.EVENT_LBUTTONDOWN, cv.EVENT_RBUTTONDOWN]:
            self.drawing = int(event == cv.EVENT_LBUTTONDOWN)
            self._draw(x, y) 
        elif event == cv.EVENT_MOUSEMOVE and self.drawing >= 0:
            self._draw(x, y)
        elif event in [cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP]:
            self.drawing = -1

    def _save(self):
        path = f'chopped/{self.name}_CHOPPED_.png'
        # convert to 4-channel
        result = cv.cvtColor(self.result, cv.COLOR_BGR2BGRA)
        print(f'saving result: {result.shape=}')
        cv.imwrite(path, self.result)

    def run(self) -> bool:
        cv.namedWindow('RESULT')
        cv.moveWindow('RESULT', 100, 0)
        cv.namedWindow('SOURCE')
        cv.moveWindow('SOURCE', 0, 0) 
        cv.setMouseCallback('SOURCE', self._handle_mouse) 
        running = True
        while running:
            cv.imshow('RESULT', self.result_view)
            cv.imshow('SOURCE', self.source_view)
            key = cv.waitKey(1)
            if   key == 27: running = False     # esc: quit
            elif key == 32: self._cut()         # spc: do grabcut
            elif key == ord('s'): self._save()
            elif key == ord('c'): self._clear_draws()
            elif key == ord('r'):               # restart
                cv.destroyAllWindows()
                return True
            elif key == ord('z'): # and cv.EVENT_FLAG_CTRLKEY: # how to check for kmod ctrl??? -> self._undo_cut()
                print('z')
            elif key == ord('z') & cv.EVENT_FLAG_CTRLKEY:
                print('ctrl-z')
                # self._undo_cut()
        cv.destroyAllWindows()
        return False

if __name__ == '__main__':
    sources = read_sources()
    image, name = ImagePicker(sources).run()
    repeat = GrabCutter(image, name).run() if image is not None else False
    while repeat:
        image, name = ImagePicker(sources).run()
        repeat = GrabCutter(image, name).run() if image is not None else False 
    print('Done')