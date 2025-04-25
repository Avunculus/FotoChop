import cv2 as cv
import os
import numpy as np
from chop import square_frame
from constants import COLORS

class ImagePicker:
    def __init__(self):
        self.window = np.zeros((864, 1536, 3), dtype=np.uint8)
        self.thumbs = {}
        self.file_name = ''
    def handle_mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN: # back-map clicks to get ix -> self.file_name = 
            col = x // 96
            row = y // 96
            ix = col + 96 * row
            self.file_name = ''
            if ix < len(self.thumbs):
                self.file_name = list(self.thumbs.keys())[ix]
            self.draw_window()
    def draw_window(self):
        for i, (name, thumb) in enumerate(self.thumbs.items()):
            if i >= 144: continue
            x = (i % 16) * 96
            y = (i // 16) * 96
            self.window[y + 4: y + 92, x + 4: x + 92, :] = thumb
            if name == self.file_name:
                cv.rectangle(self.window, (x + 6, y + 6), (x + 90, y + 90), COLORS[1], 2)
    def run(self) -> str:
        names  = [n for n in os.listdir('sources') if '.' in n]
        images = [cv.imread('sources/' + n) for n in names]
        if not images:
            print ('OOPS. Couldn\'t read any files in \'sources/\'')
            return ''
        for name, image in zip(names, images):
            if image is not None:
                match image.shape[2]:   # set color format to 3-CHANNEL
                    case 1: image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)  # grayscale
                    case 4: image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)  # 4-channel
                # 1536 x 864 == 96 * (16 x 9) // 4 + 80 + 4 = 96
                image, _ = square_frame(image, 80)
                self.thumbs[name] = cv.copyMakeBorder(
                    image, 4, 4, 4, 4, cv.BORDER_CONSTANT, value=COLORS[2]
                    )
        self.draw_window()
        cv.namedWindow('PICK IMAGE')
        cv.setMouseCallback('PICK IMAGE', self.handle_mouse)
        while True:
            cv.imshow('PICK IMAGE', self.window)
            key = cv.waitKey(1)
            if key == 27:   return ''   # esc
            elif key in [13, 32]: break   # enter|spc
        return self.file_name



def save_render(image:np.ndarray, path:str) -> str:
    if not 'renders' in os.listdir(path):
        os.mkdir(path + 'renders')
    reserved = [n.split('.') for n in os.listdir(path + 'renders/')]
    for i in range(1000):
        if repr(i) not in reserved:
            name = path + 'renders/' + repr(i) + '.png'
    cv.imwrite(name, image) # params=[cv2.IMWRITE_PNG_COMPRESSION, 0] # Set PNG compression level to 0 (no compression) thru 9 (max compress)
    return name

def write_segments(path:str, segments:list[np.ndarray]) -> None:
    # delete old masks
    for fn in [p for p in os.listdir(path) if '.' in p]:
        os.remove(path + fn)
    for i, seg in enumerate(segments):
        cv.imwrite(path + repr(i) + '.jpg', seg)
