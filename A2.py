__author__ = 'Myriam'
import math
import cv2 as cv
import numpy as np
import scipy.ndimage as ndi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def gder(img, sigma, n):
    rd, gd, bd = (0.0,) * 3
    red = img[:, :, 2]
    green = img[:, :, 1]
    blue = img[:, :, 0]

    if n is 1:  # first order derivative
        # red
        rdx = ndi.gaussian_filter(red, sigma, order=[1, 0], output=np.float64, mode='nearest')
        rdy = ndi.gaussian_filter(red, sigma, order=[0, 1], output=np.float64, mode='nearest')
        rd = np.sqrt(rdx**2 + rdy**2)

        # green
        gdx = ndi.gaussian_filter(green, sigma, order=[1, 0], output=np.float64, mode='nearest')
        gdy = ndi.gaussian_filter(green, sigma, order=[0, 1], output=np.float64, mode='nearest')
        gd = np.sqrt(gdx**2 + gdy**2)

        # blue
        bdx = ndi.gaussian_filter(blue, sigma, order=[1, 0], output=np.float64, mode='nearest')
        bdy = ndi.gaussian_filter(blue, sigma, order=[0, 1], output=np.float64, mode='nearest')
        bd = np.sqrt(bdx**2 + bdy**2)

    if n is 2:  # second order derivative
        rdx = ndi.gaussian_filter(red, sigma, order=[2, 0], output=np.float64, mode='nearest')
        rdy = ndi.gaussian_filter(red, sigma, order=[0, 2], output=np.float64, mode='nearest')
        rdxy = ndi.gaussian_filter(red, sigma, order=[1, 1], output=np.float64, mode='nearest')
        rd = np.sqrt(rdx**2 + 4*rdxy**2 + rdy**2)

        # green
        gdx = ndi.gaussian_filter(green, sigma, order=[2, 0], output=np.float64, mode='nearest')
        gdy = ndi.gaussian_filter(green, sigma, order=[0, 2], output=np.float64, mode='nearest')
        gdxy = ndi.gaussian_filter(green, sigma, order=[1, 1], output=np.float64, mode='nearest')
        gd = np.sqrt(gdx**2 + 4*gdxy**2 + gdy**2)

        # blue
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

    size = float(img.shape[0] * img.shape[1])
    kb = math.pow(kb / size, 1./float(p))
    kg = math.pow(kg / size, 1./float(p))
    kr = math.pow(kr / size, 1./float(p))

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

def maxRGB(img):
    kb, kg, kr = (0.0,) * 3

    kb = np.max(img[:, :, 0])
    kg = np.max(img[:, :, 1])
    kr = np.max(img[:, :, 2])

    for y in range(0, img.shape[1]):
        for x in range(0, img.shape[0]):
            img[x, y, 0] /= kb
            img[x, y, 1] /= kg
            img[x, y, 2] /= kr

def gamut(img):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    red = img[:, :, 2]
    green = img[:, :, 1]
    blue = img[:, :, 0]


    bgr = np.zeros(img.shape)
    cv.normalize(img, bgr, 0, 1, cv.NORM_MINMAX)
    size = img.shape[0] * img.shape[1]
    bgr = np.reshape(bgr, [size, 3])

    ax.scatter(blue, green, red, marker='o', facecolors=bgr)

    ax.set_xlabel('Blue')
    ax.set_ylabel('Green')
    ax.set_zlabel('Red')

    plt.show()


def main():
    filename = "white_balance_example_color_checkers.jpg"

    img1 = cv.imread(filename)
    img1 = img1.astype(float)

    gamut(img1)

    isMaxRGB = False
    isGreyWorld = False
    isGreyEdge = True
    normalize = True

    if isGreyWorld:
        greyworld(img1)
    elif isGreyEdge:
        sigma = 2
        p = 2
        n = 2
        greyedge(img1, sigma, n, p)
    elif isMaxRGB:
        maxRGB(img1)

    if normalize:
        cv.normalize(img1, img1, 0, 1, cv.NORM_MINMAX)

    cv.imshow('dst_rt', img1)
    cv.waitKey(0)

if __name__ == '__main__':
    main()
