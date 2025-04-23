import cv2 as cv

def main():
    global SRC
    SRC = cv.resize(cv.imread('assets/link.jpg'), (600, 400))
    kernel_max = 11
    global kbar_title
    kbar_title = 'Kernel siz'
    cv.namedWindow('ERODE')
    cv.createTrackbar(kbar_title, 'ERODE', 0, kernel_max, erode)
    cv.imshow('ERODE', SRC)
    while True:
        key = cv.waitKey(1)
        if key == 27: break
    cv.destroyAllWindows()

def erode(size):
    shape = cv.MORPH_ERODE
    element = cv.getStructuringElement(shape, (2 * size + 1, 2 * size + 1), (size, size))
    img = cv.erode(SRC, element)
    cv.imshow('ERODE', img)


if __name__ == "__main__":
    main()
    quit()