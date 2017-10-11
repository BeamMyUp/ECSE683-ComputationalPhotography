__author__ = 'Myriam'
import math
import cv2 as cv
import numpy as np
import scipy.ndimage as ndi

def gder(img, sigma, n):
    rd, gd, bd = (0.0,) * 3

    if n is 1:  # first order derivative
        # red
        red = img[:, :, 2]
        rdx = ndi.gaussian_filter(red, sigma, order=[1, 0], output=np.float64, mode='nearest')
        rdy = ndi.gaussian_filter(red, sigma, order=[0, 1], output=np.float64, mode='nearest')
        rd = np.sqrt(rdx**2 + rdy**2)

        # green
        green = img[:, :, 1]
        gdx = ndi.gaussian_filter(green, sigma, order=[1, 0], output=np.float64, mode='nearest')
        gdy = ndi.gaussian_filter(green, sigma, order=[0, 1], output=np.float64, mode='nearest')
        gd = np.sqrt(gdx**2 + gdy**2)

        # blue
        blue = img[:, :, 0]
        bdx = ndi.gaussian_filter(blue, sigma, order=[1, 0], output=np.float64, mode='nearest')
        bdy = ndi.gaussian_filter(blue, sigma, order=[0, 1], output=np.float64, mode='nearest')
        bd = np.sqrt(bdx**2 + bdy**2)

    if n is 2:  # second order derivative
        red = img[:, :, 2]
        rdx = ndi.gaussian_filter(red, sigma, order=[2, 0], output=np.float64, mode='nearest')
        rdy = ndi.gaussian_filter(red, sigma, order=[0, 2], output=np.float64, mode='nearest')
        rdxy = ndi.gaussian_filter(red, sigma, order=[1, 1], output=np.float64, mode='nearest')
        rd = np.sqrt(rdx**2 + 4*rdxy**2 + rdy**2)

        # green
        green = img[:, :, 1]
        gdx = ndi.gaussian_filter(green, sigma, order=[2, 0], output=np.float64, mode='nearest')
        gdy = ndi.gaussian_filter(green, sigma, order=[0, 2], output=np.float64, mode='nearest')
        gdxy = ndi.gaussian_filter(green, sigma, order=[1, 1], output=np.float64, mode='nearest')
        gd = np.sqrt(gdx**2 + 4*gdxy**2 + gdy**2)

        # blue
        blue = img[:, :, 0]
        bdx = ndi.gaussian_filter(blue, sigma, order=[2, 0], output=np.float64, mode='nearest')
        bdy = ndi.gaussian_filter(blue, sigma, order=[0, 2], output=np.float64, mode='nearest')
        bdxy = ndi.gaussian_filter(blue, sigma, order=[1, 1], output=np.float64, mode='nearest')
        bd = np.sqrt(bdx**2 + 4*bdxy**2 + bdy**2)

    return bd, gd, rd

def greyedge(img, sigma, n, p):
    kb, kg, kr = (0.0,) * 3
    bd, gd, rd = gder(img, sigma, n)

    bd = np.abs(bd**p)
    gd = np.abs(gd**p)
    rd = np.abs(rd**p)

    for y in range(0, img.shape[1]):
        for x in range(0, img.shape[0]):
            kb += bd[x, y]
            kg += gd[x, y]
            kr += rd[x, y]

    size = img.shape[0] * img.shape[1]
    kb = math.pow(kb / size, 1/p)
    kg = math.pow(kg / size, 1/p)
    kr = math.pow(kr / size, 1/p)

    for y in range(0, img.shape[1]):
        for x in range(0, img.shape[0]):
            img[x, y, 0] /= kb
            img[x, y, 1] /= kg
            img[x, y, 2] /= kr


def greyworld(img):
    kb, kg, kr = (0.0,) * 3

    for y in range(0, img.shape[1]):
        for x in range(0, img.shape[0]):
            kb += img[x, y, 0]
            kg += img[x, y, 1]
            kr += img[x, y, 2]

    size = img.shape[0] * img.shape[1]

    kb /= size
    kg /= size
    kr /= size

    for y in range(0, img.shape[1]):
        for x in range(0, img.shape[0]):
            img[x, y, 0] /= kb
            img[x, y, 1] /= kg
            img[x, y, 2] /= kr


def main():
    filename = "white_balance_example_color_checkers.jpg"

    img1 = cv.imread(filename)
    img1 = img1.astype(float)

    isGreyWorld = False
    isGreyEdge = True
    normalize = True

    if isGreyWorld:
        greyworld(img1)
    elif isGreyEdge:
        param = 3 # 6
        greyedge(img1, 3, 1, 2)

    if normalize:
        cv.normalize(img1, img1, 0, 1, cv.NORM_MINMAX)

    cv.imshow('dst_rt', img1)
    cv.waitKey(0)

if __name__ == '__main__':
    main()
