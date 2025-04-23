from constants import *
PX_MAX = 1000 * 1000

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

def get_chop_window(source:np.ndarray)-> np.ndarray:
        win, _ = square_frame(source, SIDE)
        win = cv.copyMakeBorder(win, 0, 0, BAR, 0, cv.BORDER_CONSTANT, value=COLORS[3])
        return win

def scale_max_pixels(source:np.ndarray, px_max) -> np.ndarray:
    h, w = source.shape[:2]
    while h * w > px_max:
        h *= .95
        w *= .95
    h, w = (round(h), round(w))
    print(f'resized from:\n{source.shape[:2]} [{source.shape[0] / source.shape[1]}] \nto: \n({h}, {w}) [{h / w}]')
    return cv.resize(source, (w, h))


class Chopper:
    def __init__(self, source:np.ndarray):
        self.source = source
        self.src_view, _ = square_frame(source, SIDE, False)
        self.win = get_chop_window(source)
        self.gc_source = scale_max_pixels(source, PX_MAX) # SIDE->GC_SIDE?Biggercanvas
        h, w = self.gc_source.shape[:2]
        self.gc_mask = np.zeros((h, w))
        self.bgm = np.zeros((1, 65), np.float64)    # background model
        self.fgm = np.zeros((1, 65), np.float64)    # foreground model

    def show_preview(self):
        bitmask = np.where((self.gc_mask==2)|(self.gc_mask==0), 0, 1).astype('uint8')
        preview = self.gc_source * bitmask[:, :, np.newaxis]
        preview = cv.resize(preview, (self.src_view.shape[1], self.src_view.shape[0]))
        cv.imshow('PREVIEW', preview)
    
    def cut(self) -> None:
        self.mask_bkup = self.gc_mask.copy()
        self.gc_mask, self.bgm, self.fgm = \
            cv.grabCut(self.gc_source, self.gc_mask, None, self.bgm, self.fgm, 1, cv.GC_INIT_WITH_MASK)
        self.show_preview()
        
    def undo_cut(self) -> None:
        self.gc_mask = self.mask_bkup.copy()
        self.show_preview()

    def draw(self) -> None: # !!! move this to handle mouse!!!
        ... # draw to gc mask -> needs scale from src_view to gc_rez

    def undo_draw(self) -> None:
        ...

    def handle_mouse(self, event:int, x:int, y:int, flags:int, param):
        if event == cv.EVENT_LBUTTONDOWN:
            ... # mousedown -> save copy of mask for undo draw

    def run(self) -> np.ndarray|None:
        # get ROI: scaled to grabcut rez
        h, w = self.gc_source.shape[:2]
        roi = (0, 0, 0, 0)
        roi = cv.selectROI('select ROI, then press spacebar', self.gc_source)
        if not all(roi[2:]): roi = (0, 0, w, h)
        cv.destroyWindow('select ROI, then press spacebar')
        # first cut: init with rect
        self.gc_mask, self.bgm, self.fgm = \
            cv.grabCut(self.gc_source, self.gc_mask, roi, self.bgm, self.fgm, 1, cv.GC_INIT_WITH_RECT)
        cv.namedWindow('CHOPPER')
        cv.setMouseCallback('CHOPPER', self.handle_mouse)
        cv.imshow('CHOPPER', self.win)
        cv.namedWindow('PREVIEW')
        self.show_preview()



class Finisher:
    def __init__(self, mask:np.ndarray):
        self.mask = mask
    def erode(self):
        ...
    def dilate(self):
        ...
    def run(self) -> np.ndarray:
        return self.mask

###################################################################################################
###################################################################################################


# BRUSHES = {0: {'view': BLACK, 'mask': cv.GC_BGD},       # background
#            1: {'view': WHITE, 'mask': cv.GC_FGD}}       # foreground
# # 2 == cv.GC_PR_BGD; 3 == cv.GC_PR_FGD

    
# def read_sources() -> dict[str,np.ndarray]:
#     """Returns {filename: thumbnail image} for all images in 'source images/' """
#     thumbs = [n.removesuffix('.jpg') for n in os.listdir('source images/thumbnails/')]
#     fnames = [n for n in os.listdir('source images/') if '.' in n]
#     for fn in fnames:           # 'xyz.jpg'
#         name = fn.split('.')[0]
#         if name not in thumbs:
#             img = cv.imread('source images/' + fn)
#             thumb, _ = scaledown_fit(img, (64, 64))
#             cv.imwrite('source images/thumbnails/' + name + '.jpg', thumb)
#     sources = {}
#     for fn in fnames:
#         name = fn.split('.')[0]
#         sources[fn] = cv.imread('source images/thumbnails/' + name + '.jpg')
#     return sources

# class GrabCutter:
#     def __init__(self, image: np.ndarray, job_name: str):
#         print(f'SOURCE:\n{job_name=}\n{image.shape=}')      # image coming in as 3-channel BGR
#         view_img, view_scale = scaledown_fit(image, (W, H)) # scale full image for viewing 
#         ...
#         self.cut_count =  0
#         self.drawing   = -1       # 0 = drawing bg, 1 = drawing fg
#         self.draw_rad  =  3       # brush radius
#         self.mask_pre_cut  = self.gc_mask.copy()    # for undoing cuts
#         self.mask_post_cut = self.gc_mask.copy()    # for clearing draws
#         self.bitmask = cv.resize(np.where((gc_mask==2)|(gc_mask==0), 0, 255).astype('uint8'), 
#                                  (self.source_view.shape[1], self.source_view.shape[0]))

#     def _refresh_views(self):
#         h, w = self.result_view.shape[:2]
#         self.result_view = cv.resize(self.result.copy(), (w, h))
#         self.bitmask = cv.resize(self.bitmask, (w, h))
#         self.source_view = self.source_view_clean.copy()

#     def _draw(self, x, y):
#         # draw to view, scaled up draw to mask: (x|y|w|h * view_scale) // gc_scale
#         cv.circle(self.source_view, (x, y), self.draw_rad, BRUSHES[self.drawing]['view'], -1)
#         x0, y0, rad = [(n * self.view_scale) // self.gc_scale for n in (x, y, self.draw_rad)]
#         cv.circle(self.gc_mask, (x0, y0), rad, BRUSHES[self.drawing]['mask'], -1)

#     def _handle_mouse(self, event, x, y, flags, *args):
#         if event in [cv.EVENT_LBUTTONDOWN, cv.EVENT_RBUTTONDOWN]:
#             self.drawing = int(event == cv.EVENT_LBUTTONDOWN)
#             self._draw(x, y) 
#         elif event == cv.EVENT_MOUSEMOVE and self.drawing >= 0:
#             self._draw(x, y)
#         elif event in [cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP]:
#             self.drawing = -1
#
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


# def get_job_names() -> list[str]:
#     return [n.split('.')[0] for n in os.listdir('chopped/') if '.' in n]

# def set_job_name(base_name:str, reserved:list[str]) -> str:
#     i = 0
#     name = base_name
#     while name in reserved:
#         i += 1
#         name = base_name + '_' + repr(i)
#     return name
