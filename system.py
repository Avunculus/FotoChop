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
