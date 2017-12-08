# This file is part of tdsi-deep-echo-challenge
import matplotlib.pyplot as plt
import numpy as np


def plotCenterOrientation(image, center, orientation):
    """
    This function plots the center and the main orientation of the region contained in the binary image
    :param image: Binary image containing one region of interest (numpy 2D array of 1' & 0')
    :param center: a tuple containing (row, column) coordinates for the center
    :param orientation: a tuple containing x- & y-eigenvalues of the region (correspond to the main orientation)
    :return: displays a matplotlib.pyplot with the given info.

    Use : plotCenterOrientation(image, (row, col), (x, y))
    """
    scale = 35
    plt.imshow(image, cmap='Greys_r')
    plt.plot([center[1] - orientation[0] * scale, center[1] + orientation[0] * scale],
             [center[0] - orientation[1] * scale, center[0] + orientation[1] * scale], color='red')

    circle = plt.Circle((center[1], center[0]), 1, color='blue')
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_artist(circle)
    plt.axis('equal')
    plt.title('Center & main orientation of the left ventricle')
    plt.show()


def plotImageMask(image, mask, phase='ED'):
    """
    This function plots an ultrasound image and its associated mask ('ground truth')
    :param image: the ultrasound image (as a numpy 2D array)
    :param mask: the ground truth mask (as a numpy 2D array)
    :param phase: string specifying 'ED' or 'ES' for the phase
    :return: a matplotlib.pyplot
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image, cmap='Greys_r')
    ax1.set_title('{} image'.format(phase))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(mask, cmap='Greys_r')
    ax2.set_title('{} Ground Truth Mask'.format(phase))
    plt.show()
