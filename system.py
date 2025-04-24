import cv2 as cv
import os
from numpy import ndarray as array

def write_segments(path:str, segments:list[array]) -> None:
    # delete old masks
    for fn in os.listdir(path):
        os.remove(path + fn)
    for i, seg in enumerate(segments):
        cv.imwrite(path + repr(i) + '.jpg', seg)


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

