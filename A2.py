__author__ = 'Myriam'
import math
import cv2 as cv
import numpy as np
import scipy.ndimage as ndi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def gder(img, sigma, n):
    rd, gd, bd = (0.0,) * 3
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

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

    return rd, gd, bd

def greyedge(img, sigma, n, p):
    kr, kg, kb = (0.0,) * 3
    rd, gd, bd = gder(img, sigma, n)

    rd = np.abs(rd**p)
    gd = np.abs(gd**p)
    bd = np.abs(bd**p)

    for y in range(0, img.shape[1]):
        for x in range(0, img.shape[0]):
            kr += rd[x, y]
            kg += gd[x, y]
            kb += bd[x, y]

    size = float(img.shape[0] * img.shape[1])
    kr = math.pow(kr / size, 1./float(p))
    kg = math.pow(kg / size, 1./float(p))
    kb = math.pow(kb / size, 1./float(p))

    for y in range(0, img.shape[1]):
        for x in range(0, img.shape[0]):
            img[x, y, 0] /= kr
            img[x, y, 1] /= kg
            img[x, y, 2] /= kb


def greyworld(img):
    kr, kg, kb = (0.0,) * 3

    for y in range(0, img.shape[1]):
        for x in range(0, img.shape[0]):
            kr += img[x, y, 0]
            kg += img[x, y, 1]
            kb += img[x, y, 2]

    size = img.shape[0] * img.shape[1]

    kr /= size
    kg /= size
    kb /= size

    for y in range(0, img.shape[1]):
        for x in range(0, img.shape[0]):
            img[x, y, 0] /= kr
            img[x, y, 1] /= kg
            img[x, y, 2] /= kb


def maxRGB(img):
    kr = np.max(img[:, :, 0])
    kg = np.max(img[:, :, 1])
    kb = np.max(img[:, :, 2])

    for y in range(0, img.shape[1]):
        for x in range(0, img.shape[0]):
            img[x, y, 0] /= kr
            img[x, y, 1] /= kg
            img[x, y, 2] /= kb

def gamut(img, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    rgb = np.zeros(img.shape)
    cv.normalize(img, rgb, 0, 1, cv.NORM_MINMAX)
    size = img.shape[0] * img.shape[1]
    rgb = np.reshape(rgb, [size, 3])

    ax.scatter(red, green, blue, marker='o', facecolors=rgb)

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    ax.view_init(elev=20., azim=-135)
    plt.savefig(filename)


def main():
    filename = "imgA2/framed_sm.jpg"

    img1 = cv.imread(filename)
    b, g, r = cv.split(img1)
    img1 = cv.merge([r, g, b])
    img1 = img1.astype(float)

    cv.normalize(img1, img1, 0, 1, cv.NORM_MINMAX)
    # gamut(img1, "framed_gamut.jpg")

    isMaxRGB = False
    isGreyWorld = True
    isGreyEdge = False
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

    gamut(img1, "framed_gamut_greyWorld.jpg")

    r, g, b = cv.split(img1)
    img1 = cv.merge([b, g, r])
    cv.imshow('dst_rt', img1)
#
    cv.normalize(img1, img1, 0, 255, cv.NORM_MINMAX)
    cv.imwrite("framed_greyEdge.jpg", img1)
    cv.waitKey(0)

if __name__ == '__main__':
    main()
