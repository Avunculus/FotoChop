Cut an object from the background in an image file, save the object as 4 channel image with transparent bg.

# Usage
1. move source image(s) to 'source images/'
2. run chop.py 
3. choose a source, then press spacebar
4. click-drag to select the ROI (region of interest; the object's bounding rectangle), then press spacebar
5. use mouse and keys to remove the background:
- use mouse buttons to indicate areas that are **definite foreground (left-click)** or **definite background (right-click).**
- press spacebar at any time, with or without additional draws, to call one iter of cv.grabcut().
- the number of iters needed to get clean edges varies; I find 3-6 iters is usually good.
6. save the result by pressing 's'

# Keymap
|**input**|**command**|
|--------:|:----------|
|escape   |quit |
|spacebar |single iter cv.grabcut() / accept selection |
|l-click  |draw: definite foreground / make selection |
|r-click  |draw: definite background |
|ctrl-z   |undo most recent cut |
|c        |clear all draws, undo all cuts (keeps ROI) |
|s        |save object image to 'segments/___.png' |
|r        |restart (choose new image) |