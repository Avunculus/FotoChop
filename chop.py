from constants import *

def square_frame(image:np.ndarray, side:int, pad=True)-> np.ndarray:
    """Returns image resized to fit long side into square frame.
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
    return image

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

class Chopper:
    def __init__(self, source:np.ndarray):
        self.src_full = source
        self.src_scaled = square_frame(self.src_full, SIDE, False)# SIDE->GC_SIDE?Biggercanvas
        # get ROI
        h, w = self.src_scaled.shape[:2]
        roi = (0, 0, 0, 0)
        roi = cv.selectROI('select ROI, then press spacebar', self.src_scaled)
        if not all(roi[2:]): roi = (0, 0, w, h)
        cv.destroyWindow('select ROI, then press spacebar')
        # first cut: init with rect
        self.gc_mask = np.zeros((h, w))
        self.bgm = np.zeros((1, 65), np.float64)    # background model: init from full img on first cut
        self.fgm = np.zeros((1, 65), np.float64)    # foreground model: init from full img on first cut
        self.gc_mask, self.bgm, self.fgm = \
            cv.grabCut(self.src_scaled, self.gc_mask, roi, self.bgm, self.fgm, 1, cv.GC_INIT_WITH_RECT)
        self.bitmask = np.where((self.gc_mask==2)|(self.gc_mask==0), 0, 1).astype('uint8')
        self.preview = self.src_scaled * self.bitmask[:, :, np.newaxis]
        cv.namedWindow('PREVIEW')
        cv.imshow('PREVIEW', self.preview)
        

    def draw_win(self) -> np.ndarray:
        win = square_frame(self.src_scaled, SIDE)
        win = cv.copyMakeBorder(win, 0, 0, BAR, 0, cv.BORDER_CONSTANT, value=COLORS[0])
        # cv.namedWindow('TEST'); cv.imshow('TEST', win)
        return win

    
    def handle_mouse(self, event:int, x:int, y:int, flags:int, param):
        ...

    def cut(self) -> None:
        ...
    def undo_cut(self) -> None:
        ...
    def draw(self) -> None:
        ...
    def undo_draw(self) -> None:
        ...

    def run(self) -> np.ndarray|None:
        cv.namedWindow('CHOPPER')
        cv.setMouseCallback('CHOPPER', self.handle_mouse)
        self.win = self.draw_win()
        cv.imshow('CHOPPER', self.win)




###################################################################################################
###################################################################################################
# class Segmentor:
#     def __init__(self):
#         self.window = np.zeros((SIDE, 2 * SIDE + BAR, 3))
#         # draw sidebar ui
#         ...
#     def handle_mouse(self, event:int, x:int, y:int, flags:int, param):
#         if event == cv.EVENT_LBUTTONDOWN:
#             print(f'clicked \'SEGMENT\' window @ ({x}, {y})')
#     def get_source(self) -> np.ndarray:
#         return slice_rect(SOURCE, self.rect)
#     def run(self):
#         self.rect = get_roi(SOURCE) # !! getting zero-dim error for w|h...
#         ... # check dims for 0-width|hgt? *** select_roi needs fixing. Thread issue.
#         self.source = self.get_source()
#         cv.namedWindow('SEGMENTOR')
#         cv.setMouseCallback('SEGMENTOR', self.handle_mouse)
#         cv.imshow('SEGMENTOR', self.window)
###################################################################################################
###################################################################################################


# BRUSHES = {0: {'view': BLACK, 'mask': cv.GC_BGD},       # background
#            1: {'view': WHITE, 'mask': cv.GC_FGD}}       # foreground
# # 2 == cv.GC_PR_BGD; 3 == cv.GC_PR_FGD

# def scaledown_fit(img: np.ndarray, limit: int|tuple[int,int]) -> tuple[np.ndarray,int]:
#     """Integer scale image down to fit shape: (w, h) or size: pixels. Keeps aspect ratio."""
#     assert isinstance(limit, tuple) or isinstance(limit, int), f'bad limit dtype: {limit}'
#     h, w = img.shape[:2]
#     scale = 1
#     while (isinstance(limit, int) and (w * h) // (scale * scale) > limit) or \
#         (isinstance(limit, tuple) and (w / scale > limit[0] or h / scale > limit[1])):
#         scale += 1
#     shape = (w // scale, h // scale)
#     img = cv.resize(img, shape)
#     print(f'image scaled: {scale=}')
#     return (img, scale)
    
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

# def first_cut(gc_mask: np.ndarray, source: np.ndarray) -> np.ndarray:
#     """Removes pixels marked BG/PR_BG in the gc_mask from source. Scales gc_mask up to source."""
#     mask = np.where((gc_mask==2)|(gc_mask==0), 0, 1).astype('uint8')
#     h, w = source.shape[:2]
#     mask = cv.resize(mask, (w, h))
#     return source * mask[:, :, np.newaxis] 

# class GrabCutter:
#     def __init__(self, image: np.ndarray, job_name: str):
#         print(f'SOURCE:\n{job_name=}\n{image.shape=}')      # image coming in as 3-channel BGR
#         view_img, view_scale = scaledown_fit(image, (W, H)) # scale full image for viewing 
#         roi = (0, 0, 0, 0)              # get region of interest from full image (x, y, w, h)
#         while any([i == 0 for i in roi[2:]]):
#             roi = cv.selectROI('select ROI, then press spacebar', view_img)
#         cv.destroyWindow('select ROI, then press spacebar')
#         # scale image, rect for first call to cv.grabcut()
#         gc_img, gc_scale = scaledown_fit(image, MAX_PXL)
#         x, y, w, h = [(n * view_scale) // gc_scale for n in roi]
#         roi = (x, y, w, h)
#         # FIRST CUT: init with rect -> crop & rescale
#         gc_mask = np.zeros(gc_img.shape[:2])
#         bgm = np.zeros((1, 65), np.float64)    # background model: init from full img on first cut
#         fgm = np.zeros((1, 65), np.float64)    # foreground model: init from full img on first cut
#         gc_mask, bgm, fgm = cv.grabCut(gc_img, gc_mask, roi, bgm, fgm, 1, cv.GC_INIT_WITH_RECT)
#         result = first_cut(gc_mask, image)
#         gc_mask = gc_mask[y: y + h, x: x + w]
#         x, y, w, h = [n * gc_scale for n in roi] # scale up to full
#         # set job attributes
#         self.job_name = job_name
#         self.source = image[y: y + h, x: x + w]; print(f'ROI selected: {self.source.shape=}')
#         self.source_view, self.view_scale = scaledown_fit(self.source.copy(), (W, H))
#         self.source_view_clean = self.source_view.copy()   # not drawn to
#         self.gc_source, self.gc_scale = scaledown_fit(self.source.copy(), MAX_PXL)
#         self.gc_mask = cv.resize(gc_mask, (self.gc_source.shape[1], self.gc_source.shape[0]))
#         self.bgm = bgm
#         self.fgm = fgm
#         self.result = result[y: y + h, x: x + w]
#         self.result_view = cv.resize(self.result.copy(),
#                                      (self.source_view.shape[1], self.source_view.shape[0]))
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
        
#     def _cut(self):
#         print('cutting...', end='')
#         self.mask_pre_cut = self.gc_mask.copy()
#         self.gc_mask, self.bgm, self.fgm = \
#             cv.grabCut(self.gc_source, self.gc_mask, None,
#                        self.bgm, self.fgm, 1, cv.GC_INIT_WITH_MASK)
#         self.mask_post_cut = self.gc_mask.copy()
#         self.cut_count += 1
#         print(f' cut # {self.cut_count} complete.\tpixels: [add bitmask.nonzero() for pixel counts...]')
#         self._apply_mask()

#     def _apply_mask(self, from_gc_mask:bool=True) -> None:
#         mask = np.where((self.gc_mask==2)|(self.gc_mask==0), 0, 1).astype('uint8')\
#             if from_gc_mask else self.bitmask // 255
#         h, w = self.source.shape[:2]
#         mask = cv.resize(mask, (w, h))
#         self.result = self.source * mask[:, :, np.newaxis]
#         if from_gc_mask: self.bitmask = cv.resize(mask.copy(), (w, h)) * 255 # update bitmask if cutting from cv.grabut()
#         self._refresh_views()

#     def _undo_cut(self):
#         if self.cut_count < 1: return
#         self.gc_mask = self.mask_pre_cut.copy()
#         self.cut_count -= 1
#         self._refresh_views()
#     def _draw(self, x, y):
#         # draw to view, scaled up draw to mask: (x|y|w|h * view_scale) // gc_scale
#         cv.circle(self.source_view, (x, y), self.draw_rad, BRUSHES[self.drawing]['view'], -1)
#         x0, y0, rad = [(n * self.view_scale) // self.gc_scale for n in (x, y, self.draw_rad)]
#         cv.circle(self.gc_mask, (x0, y0), rad, BRUSHES[self.drawing]['mask'], -1)
#     def _clear_draws(self):
#         self.gc_mask = self.mask_post_cut.copy()
#         self.source_view = self.source_view_clean.copy()
#     def _handle_mouse(self, event, x, y, flags, *args):
#         if event in [cv.EVENT_LBUTTONDOWN, cv.EVENT_RBUTTONDOWN]:
#             self.drawing = int(event == cv.EVENT_LBUTTONDOWN)
#             self._draw(x, y) 
#         elif event == cv.EVENT_MOUSEMOVE and self.drawing >= 0:
#             self._draw(x, y)
#         elif event in [cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP]:
#             self.drawing = -1

#     def _erode_mask(self):
#         size = 3
#         element = cv.getStructuringElement(cv.MORPH_RECT, (2 * size + 1, 2 * size + 1), (size, size))
#         self.bitmask = cv.erode(self.bitmask, element)
#         ... # size, shape -> structuruing element; erode|dilate
#         self._apply_mask(from_gc_mask=False)

#     def _dilate_mask(self):
#         size = 3
#         element = cv.getStructuringElement(cv.MORPH_RECT, (2 * size + 1, 2 * size + 1), (size, size))
#         self.bitmask = cv.erode(self.bitmask, element)
#         self._apply_mask(from_gc_mask=False)

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
#     def run(self) -> bool:
#         cv.namedWindow('RESULT')
#         cv.moveWindow('RESULT', 640, 0)
#         cv.namedWindow('MASK')
#         cv.moveWindow('MASK', 320, 0)
#         cv.namedWindow('SOURCE')
#         cv.moveWindow('SOURCE', 0, 0) 
#         cv.setMouseCallback('SOURCE', self._handle_mouse) 
#         running = True
#         while running:
#             cv.imshow('RESULT', self.result_view)
#             cv.imshow('SOURCE', self.source_view)
#             cv.imshow('MASK', self.bitmask)
#             key = cv.waitKey(1)
#             if   key == 27: running = False             # esc: quit
#             elif key == 32: self._cut()                 # spc: do grabcut
#             elif key == 26: self._undo_cut()            # ctrl-z
#             elif key == ord('s'): self._save()
#             elif key == ord('c'): self._clear_draws()
#             elif key == ord('e'): self._erode_mask()
#             elif key == ord('d'): self._dilate_mask()
#             elif key == ord('r'):                       # restart
#                 cv.destroyAllWindows()
#                 return True
#         cv.destroyAllWindows()
#         return False

# def get_job_names() -> list[str]:
#     return [n.split('.')[0] for n in os.listdir('chopped/') if '.' in n]

# def set_job_name(base_name:str, reserved:list[str]) -> str:
#     i = 0
#     name = base_name
#     while name in reserved:
#         i += 1
#         name = base_name + '_' + repr(i)
#     return name

# if __name__ == '__main__':
#     sources = read_sources()
#     image, name = ImagePicker(sources).run()
#     prev_jobs = get_job_names()
#     name = set_job_name(name, prev_jobs)
#     repeat = GrabCutter(image, name).run() if image is not None else False
#     while repeat:
#         sources = read_sources()
#         image, name = ImagePicker(sources).run()
#         prev_jobs = get_job_names()
#         name = set_job_name(name, prev_jobs)
#         repeat = GrabCutter(image, name).run() if image is not None else False
#     print('Done')