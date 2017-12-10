# This file is part of tdsi-deep-echo-challenge

import pandas as pd
import numpy as np
from scipy import ndimage, misc
import os

from . import acquisition


def getRoi(image, pixelvalue):
    """
    This function takes in an image (as a numpy 2D array), and return the regions composed of pixels with values equal
    to pixelvalue
    :param image: The image (numpy 2D array)
    :param pixelvalue: the pixel intensity value used to select the regions of interest
    :return: a numpy array of equal size to image, containing the specific region.
    """
    return ((image == pixelvalue)[:, :]).astype(int)


def resizeImgArr(img, mask, size=(96, 96)):
    """
    This function resizes an image and its associated ground truth mask to the given size.
    :param img: The original gray-level image to resize.
    :param mask: The binary mask corresponding to the main region of interest.
    :param size: a tuple corresponding to the desired new size.
    :return: the resized image and mask as numpy ndarray.
    """
    resizedImg = misc.imresize(img, size, interp='nearest')
    resizedMask = misc.imresize(mask, size, interp='nearest')

    # we work with a binary mask, which takes either 0 or 1 as pixel intensity value. misc.imresize resets the gray
    # level, fitting them between 0 & 255. We manually rearrange the value. HAS TO BE FIXED IN FUTURE COMMIT
    resizedMask[resizedMask == 255] = 1
    return resizedImg, resizedMask

def findCenter(img):
    """
    This function returns the center coordinates of the different connected regions of an image.
    :param img: input image
    :return: ([x1, x2 ... xn], [y1, y2 ... yn]) where xi,yi are the coordinates of the ith region detected in the
    image (total of n regions). If only one region is detected, the 2 coordinates are returned as a tuple (x,y).
    """
    # use a boolean condition to find where pixel values are = 1
    blobs = (img == 1)
    # label the n connected regions that satisfy this condition
    labels, nlabels = ndimage.label(blobs)
    # Find their unweighted centroids
    r, c = np.vstack(ndimage.center_of_mass(blobs, labels, np.arange(nlabels) + 1)).T  # returned as np.ndarray
    # round the values to int (since pixel coordinates)
    r = np.round(r).astype(int)
    c = np.round(c).astype(int)
    if nlabels == 1:
        return r[0], c[0]
    else:
        return r.tolist(), c.tolist()


def findMainOrientation(img, pixelvalue):
    """
    This function returns the main orientation of the region composed of pixels of the specified value.
    :param img: input image, pixelvalue: the value used to filter the pixels
    :return: the x- & y-eigenvalues of the region as a tuple (correspond to the main orientation, see
    https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/)
    """
    # get the indices of the pixels of value equal to pixelvalue
    y, x = np.where(img == pixelvalue)
    #  subtract mean from each dimension.
    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])
    # covariance matrix and its eigenvectors and eigenvalues
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    # sort eigenvalues in decreasing order
    sort_indices = np.argsort(evals)[::-1]
    evec1, _ = evecs[:, sort_indices]
    # eigenvector with largest eigenvalue
    x_v1, y_v1 = evec1
    return x_v1, y_v1