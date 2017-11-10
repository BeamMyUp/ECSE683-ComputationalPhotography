__author__ = 'Myriam'

import cv2 as cv

def main():
    filename = "imgA4/ISO409600_SonyA7SII_lowlight_image.jpg"

    img1 = cv.imread(filename)
    b, g, r = cv.split(img1)
    img1 = cv.merge([r, g, b])
    img1 = img1.astype(float)

    cv.normalize(img1, img1, 0, 1, cv.NORM_MINMAX)

    cv.imshow('dst_rt', img1)
#
    cv.normalize(img1, img1, 0, 255, cv.NORM_MINMAX)
    cv.imwrite("framed_greyEdge.jpg", img1)
    cv.waitKey(0)

if __name__ == '__main__':
    main()