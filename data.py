# This file is part of tdsi-deep-echo-challenge
import sys
import os
from skimage.transform import resize
import numpy as np

# Set python path to find the local deepecho package
sys.path.insert(0, 'pwd')
from deepecho import *

#   PARAMETERS  #
data_path = 'data/'  # Data path
phase = 'ES'  # select phase for create_train_data()
img_rows = 96  # New dimensions when resizing the images
img_cols = 96
#################


def create_train_data(phase='ED', img_rows=96, img_cols=96, verbose=1):
    """
    Create training data using raw data from one phase only : either 'ED' or 'ES'. See create_train_data_2() to create
    training data using raw data from both phases.
    :param phase: a string indicating which phase to select. 'ED' = diastole, 'ES' = systole.
    :param img_rows: the new x-axis dimension used to resize the images
    :param img_cols: the new y-axis dimension used to resize the images
    :param verbose: control the display of additional information.
    :return: Resized images to (img_rows, img_cols) & targets values in .npy files.
    """
    print("Creating training data using raw data from {} phase".format(phase))
    # first, get the patients directory names located in the 'data_path' folder. These names (e.g. 'patient0001') will
    # be used for indexing.
    patients = [name for name in os.listdir(os.path.join(os.curdir, data_path)) if not name.startswith('.')]
    # We sort this list to get the patients id in increasing order
    patients.sort(key=lambda s: s[-3:])  # sort according to last 3 characters

    # create an empty numpy.ndarray which will contain the images (resized to (img_rows, img_cols))
    images = np.ndarray((len(patients), img_rows, img_cols), dtype=np.uint8)

    # create a second empty numpy.ndarray which will contain the center coord. & orientation values for each patient
    # We have 4 main features : rowCenter, colCenter, xOrientation, yOrientation (stored in that order).
    targets = np.ndarray((len(patients), 4), dtype=np.float32)

    # we now go through each patient's directory :
    for idx, patient in enumerate(patients):

        # read image & mask (only focus on ED for now)
        img, _, _, _ = acquisition.load_mhd_data('{d}/{pa}/{pa}_4CH_{ph}.mhd'.format(d=data_path, pa=patient, ph=phase))
        img_mask, _, _, _ = acquisition.load_mhd_data('{d}/{pa}/{pa}_4CH_{ph}_gt.mhd'.format(d=data_path, pa=patient, ph=phase))

        # extract the region corresponding to the left ventricle in the image mask (region where pixel = 1)
        img_mask = getRoi(img_mask, 1)

        # resize the img & the mask to (img_rows, img_cols) to keep the network input manageable
        img = resize(img, (img_cols, img_rows), mode='reflect', preserve_range=True)
        img_mask = resize(img_mask, (img_cols, img_rows), mode='reflect', preserve_range=True)

        # get the center coordinates of the left ventricle (on the resized image)
        row, col = findCenter(img_mask)

        if isinstance(row, list):  # findCenter might return a list, so we ensure row, col are scalars.
            row = row[0]
            col = col[0]

        # get the orientation of the left ventricle (on the resized image)
        x_v1, y_v1 = findMainOrientation(img_mask, 1)

        # now, save the resized image to the X dataframe as a 96x96 2D-array (which will be the network input)
        images[idx] = img

        # save the center coordinates & orientation to the y dataframe (which will be the output of the network)
        targets[idx] = np.array([row, col, x_v1, y_v1])

        if verbose:
            # print some info
            print('[{}] rowCenter, colCenter = '.format(phase), row, ',', col)
            print('[{}] xOrientation, yOrientation = '.format(phase), x_v1, ',', y_v1)
            print('######### Done: {0}/{1} patients'.format(idx+1, len(patients)))

    print('Data processing done.')
    # save both ndarrays to a .npy files (for faster loading later)
    np.save('images_phase_{}.npy'.format(phase), images)
    np.save('targets_phase_{}.npy'.format(phase), targets)
    print('Saving to .npy files done.')


def create_train_data_2(img_rows=96, img_cols=96, verbose=1):
    """
    Creating training data. Using both ED & ES images. We are only stacking them together for now, and not passing
    both phases at the same time in the neural network (reading 1 image at the time).
    :param img_rows: the new x-axis dimension used to resize the images
    :param img_cols: the new y-axis dimension used to resize the images
    :param verbose: control the display of additional information.
    :return: Resized images to (img_rows, img_cols) & targets values in .npy files.
    """
    print("Creating training data using raw data from both phases")
    # first, get the patients directory names located in the 'data_path' folder. These names (e.g. 'patient0001') will
    # be used for indexing.
    patients = [name for name in os.listdir(os.path.join(os.curdir, data_path)) if not name.startswith('.')]
    # We sort this list to get the patients id in increasing order
    patients.sort(key=lambda s: s[-3:])  # sort according to last 3 characters

    # create an empty numpy.ndarray which will contain the images (resized to (img_rows, img_cols))
    images = np.ndarray((2 * len(patients), img_rows, img_cols), dtype=np.uint8)  # x 2 in len as storing ED & ES

    # create a second empty numpy.ndarray which will contain the center coord. & orientation values for each patient
    # for both phases.
    # We have 4 main features : rowCenter, colCenter, xOrientation, yOrientation (stored in that order).
    targets = np.ndarray((2 * len(patients), 4), dtype=np.float32)

    # define iterable containing the different phases
    phases = ['ED', 'ES']

    idx = 0
    # we now go through each patient's directory :
    for patient in patients:

        for phase in phases:

            # read image & mask
            img, _, _, _ = acquisition.load_mhd_data('{d}/{pa}/{pa}_4CH_{ph}.mhd'.format(d=data_path, pa=patient, ph=phase))
            img_mask, _, _, _ = acquisition.load_mhd_data('{d}/{pa}/{pa}_4CH_{ph}_gt.mhd'.format(d=data_path, pa=patient, ph=phase))

            # extract the region corresponding to the left ventricle in the image mask (region where pixel = 1)
            img_mask = getRoi(img_mask, 1)

            # resize the img & the mask to (img_rows, img_cols) to keep the network input manageable
            img = resize(img, (img_cols, img_rows), mode='reflect', preserve_range=True)
            img_mask = resize(img_mask, (img_cols, img_rows), mode='reflect', preserve_range=True)

            # get the center coordinates of the left ventricle (on the resized image)
            row, col = findCenter(img_mask)

            if isinstance(row, list):  # findCenter might return a list, so we ensure row, col are scalars.
                row = row[0]
                col = col[0]

            # get the orientation of the left ventricle (on the resized image)
            x_v1, y_v1 = findMainOrientation(img_mask, 1)

            # now, save the resized image to the X dataframe as a 96x96 2D-array (which will be the network input)
            images[idx] = img

            # save the center coordinates & orientation to the y dataframe (which will be the output of the network)
            targets[idx] = np.array([row, col, x_v1, y_v1])

            idx += 1
            if verbose:
                # print some info
                print('[{}] rowCenter, colCenter = '.format(phase), row, ',', col)
                print('[{}] xOrientation, yOrientation = '.format(phase), x_v1, ',', y_v1)

        if verbose:
            print('######### Done: {0}/{1} patients'.format(round(idx/2), len(patients)))

    print('Data processing done.')
    # save both ndarrays to a .npy files (for faster loading later)
    np.save('images_both_phases.npy', images)
    np.save('targets_both_phases.npy', targets)
    print('Saving to .npy files done.')


def load_train_data(model, data):
    """
    Loading training data & doing some additional preprocessing on it. If the indicated model is a dnn, we flatten out
    the input images.
    :param model: string to indicate the type of model to prepare the data for. Either dnn or cnn
    :param data: Indicates which data to load (i.e data from both phases or from a specific one). Either 'ED', 'ES'
    or 'both'.
    :return: images & target features as numpy arrays.
    """

    print('-' * 30)
    print('Loading & processing data for {} phase'.format(data))
    print('-' * 30)

    dataname = ''

    if data == 'ED':
        dataname = '{}_phase_ED.npy'
    elif data == 'ES':
        dataname = '{}_phase_ES.npy'
    elif data == 'both':
        dataname = '{}_both_phases.npy'

    # read in the .npy file containing the images
    images = np.load(dataname.format('images'))

    # read in the .npy file containing the target features
    targets = np.load(dataname.format('targets'))

    # scale image pixel values to [0, 1]
    print('scale pixel values to [0, 1]')
    images = images.astype(np.float32)
    images /= 255.

    # scale target center coordinates to [-1, 1] (from 0 to 95 initially)
    targets = targets.astype(np.float32)
    print('scale target coordinates to [-1, 1]')
    targets[:, 0] = (targets[:, 0] - (img_rows/2))/(img_rows/2)
    targets[:, 1] = (targets[:, 1] - (img_rows / 2)) / (img_cols / 2)

    # reshape images according to the neural network model intended to be used
    if model == 'cnn':
        print('indicated model is a cnn, reshaping images with channels first.')
        images = images.reshape(-1, 1, 96, 96)
    elif model == 'dnn':
        print('indicated model is a dnn, flattening out images.')
        images = images.reshape(images.shape[0], img_rows*img_rows)

    print('-' * 30)
    print('Loading & processing done.')
    print('-' * 30)

    return images, targets


if __name__ == '__main__':
    create_train_data(phase=phase, img_rows=img_rows, img_cols=img_cols, verbose=0)
    create_train_data_2(img_rows=img_rows, img_cols=img_cols, verbose=1)
    #X, y = load_train_data(model='dnn', data='both')
    #print("X.shape = {}".format(X.shape))
    #print("y.shape = {}".format(y.shape))
