from constants import *
import os

def read_sources() -> dict[str,np.ndarray]:
    """Returns {filename: thumbnail image} for all images in 'source images/' """
    thumbs = [n.removesuffix('.jpg') for n in os.listdir('source images/thumbnails/')]
    fnames = [n for n in os.listdir('source images/') if '.' in n]
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



# def get_job_names() -> list[str]:
#     return [n.split('.')[0] for n in os.listdir('chopped/') if '.' in n]

# def set_job_name(base_name:str, reserved:list[str]) -> str:
#     i = 0
#     name = base_name
#     while name in reserved:
#         i += 1
#         name = base_name + '_' + repr(i)
#     return name


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
