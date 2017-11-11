# This file is part of tdsi-deep-echo-challenge
import sys
import os
from skimage.transform import resize
import numpy as np

# Set python path to find the local deepecho package
sys.path.insert(0, 'pwd')
from deepecho import *

data_path = 'data/'  # Data path
img_rows = 96  # New dimensions when resizing the images
img_cols = 96


def create_train_data():
    # first, get the patients directory names located in the 'data_path' folder. These names (e.g. 'patient0001') will
    # be used for indexing.
    patients = [name for name in os.listdir(os.path.join(os.curdir, data_path)) if not name.startswith('.')]
    # We sort this list to get the patients id in increasing order
    patients.sort(key=lambda s: s[-3:])

    # create an empty numpy.ndarray which will contain the images (resized as 96x96)
    images = np.ndarray((len(patients), img_rows, img_cols), dtype=np.uint8)

    # create a second empty numpy.ndarray which will contain the center coord. & orientation values for each patient
    # We have 4 main features : rowCenter, colCenter, xOrientation, yOrientation (stored in that order).
    targets = np.ndarray((len(patients), 4), dtype=np.float32)

    i = 0
    print('-' * 30)
    print('Creating training images & targets features...')
    print('-' * 30)
    # we now go through each patient's directory :
    for patient in patients:

        # read image & mask (only focus on ED for now)
        img, _, _, _ = acquisition.load_mhd_data('{d}/{pa}/{pa}_4CH_ES.mhd'.format(d=data_path, pa=patient))
        img_mask, _, _, _ = acquisition.load_mhd_data('{d}/{pa}/{pa}_4CH_ES_gt.mhd'.format(d=data_path, pa=patient))

        # extract the region corresponding to the left ventricle in the image mask (region where pixel = 1)
        img_mask = getRoi(img_mask, 1)

        # resize the img & the mask to 96x96 to keep the network input manageable
        img = resize(img, (img_cols, img_rows), mode='reflect', preserve_range=True)
        img_mask = resize(img_mask, (img_cols, img_rows), mode='reflect', preserve_range=True)

        # get the center coordinates of the left ventricle (on the resized image)
        row, col = findCenter(img_mask)

        if isinstance(row, list):  # findCenter might return a list, so we ensure row, col are scalars.
            row = row[0]
            col = col[0]

        # get the orientation of the left ventricle (on the resized image)
        x_v1, y_v1 = findMainOrientation(img_mask, 1)

        # now, save the resized image to the X dataframe as a row vector (which will be the network input)
        images[i] = img

        # save the center coordinates & orientation to the y dataframe (which will be the output of the network)
        targets[i] = np.array([row, col, x_v1, y_v1])

        i += 1
        print('Done: {0}/{1} patients'.format(i, len(patients)))

    print('Data processing done.')
    # save both ndarrays to a .npy files (for faster loading later)
    np.save('images.npy', images)
    np.save('targets.npy', targets)
    print('Saving to .npy files done.')


def load_train_data():

    print('-' * 30)
    print('Loading & processing data...')
    print('-' * 30)

    # read in the .npy file containing the images
    images = np.load('images.npy')

    # read in the .npy file containing the target features
    targets = np.load('targets.npy')

    # scale image pixel values to [0, 1]
    images = images.astype(np.float32)
    images /= 255.

    # scale target center coordinates to [-1, 1] (from 0 to 95 initially)
    targets = targets.astype(np.float32)
    targets[:, 0] = (targets[:, 0] - (img_rows/2))/(img_rows/2)
    targets[:, 1] = (targets[:, 1] - (img_cols / 2)) / (img_cols / 2)

    images = images[..., np.newaxis]

    print('-' * 30)
    print('Loading & processing done.')
    print('-' * 30)

    return images, targets


if __name__ == '__main__':
    # create_train_data()
    X, y = load_train_data()
    print("X.shape = {}; X.min = {:.3f}; X.max = {:.3f}".format(X.shape, X.min(), X.max()))
    print("y.shape = {}; y.min = {:.3f}; y.max = {:.3f}".format(y.shape, y.min(), y.max()))
