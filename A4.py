__author__ = 'Myriam'

import cv2 as cv

def main():
    filename = "imgA4/ISO409600_SonyA7SII_lowlight_image.jpg"

    img1 = cv.imread(filename)

    q1GaussBlur = True
    q1NLM = False

    if q1GaussBlur:
        img1 = cv.GaussianBlur(img1, (15, 15), 3)
    elif q1NLM:
        img1 = cv.fastNlMeansDenoising(img1, h=17, templateWindowSize=7, searchWindowSize=21)

    cv.imshow('dst_rt', img1)
#
    cv.normalize(img1, img1, 0, 255, cv.NORM_MINMAX)
    cv.imwrite("q1_gauss_15_3.jpg", img1)
    cv.waitKey(0)

if __name__ == '__main__':
    main()