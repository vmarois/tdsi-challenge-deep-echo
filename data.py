# This file is part of tdsi-deep-echo-challenge
import sys
import os
from skimage.transform import resize, AffineTransform, warp

import numpy as np

# Set python path to find the local deepecho package
sys.path.insert(0, 'pwd')
from deepecho import *

#   PARAMETERS  #
data_path = 'data/'  # Data path
phase = 'ED'  # select phase for create_train_data()
img_rows = 128  # New dimensions when resizing the images
img_cols = 128
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

    # create a third empty numpy.ndarray to store the new pixel dimensions of the images, as they change when resizing.
    pixel_dim = np.ndarray((len(patients), 2), dtype=np.float32)

    # we now go through each patient's directory :
    for idx, patient in enumerate(patients):

        # read image & mask (only focus on ED for now)
        img, _, _, _ = acquisition.load_mhd_data('{d}/{pa}/{pa}_4CH_{ph}.mhd'.format(d=data_path, pa=patient, ph=phase))
        img_mask, _, _, _ = acquisition.load_mhd_data('{d}/{pa}/{pa}_4CH_{ph}_gt.mhd'.format(d=data_path, pa=patient, ph=phase))

        # get the new pixel dimensions due to resizing : scale factor is orig_dim / new_dim
        pixel_width = 0.154 * (float(img.shape[0]) / img_cols)
        pixel_height = 0.308 * (float(img.shape[1]) / img_rows)
        # store them
        pixel_dim[idx] = np.array([pixel_width, pixel_height])

        # extract the region corresponding to the left ventricle in the image mask (region where pixel = 1)
        img_mask = getRoi(img_mask, 1)

        # resize the img & the mask to (img_rows, img_cols) to keep the network input manageable
        img = resize(img, (img_rows, img_cols), mode='reflect', preserve_range=True)
        img_mask = resize(img_mask, (img_rows, img_cols), mode='reflect', preserve_range=True)

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
    # save all ndarrays to a .npy files (for faster loading later)
    # Create directory to store pdf files.
    directory = os.path.join(os.getcwd(), 'output/processed_data/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save('output/processed_data/images_phase_{}_{}.npy'.format(phase, img_rows), images)
    np.save('output/processed_data/targets_phase_{}_{}.npy'.format(phase, img_rows), targets)
    np.save('output/processed_data/pixel_dim_phase_{}_{}.npy'.format(phase, img_rows), pixel_dim)
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

    # create a third empty numpy.ndarray to store the new pixel dimensions of the images, as they change when resizing.
    pixel_dim = np.ndarray((2 * len(patients), 2), dtype=np.float32)

    # define iterable containing the different phases
    phases = ['ED', 'ES']

    idx = 0
    # we now go through each patient's directory :
    for patient in patients:

        for phase in phases:

            # read image & mask
            img, _, _, _ = acquisition.load_mhd_data('{d}/{pa}/{pa}_4CH_{ph}.mhd'.format(d=data_path, pa=patient, ph=phase))
            img_mask, _, _, _ = acquisition.load_mhd_data('{d}/{pa}/{pa}_4CH_{ph}_gt.mhd'.format(d=data_path, pa=patient, ph=phase))

            # get the new pixel dimensions due to resizing : scale factor is orig_dim / new_dim
            pixel_width = 0.154 * (float(img.shape[0]) / img_cols)
            pixel_height = 0.308 * (float(img.shape[1]) / img_rows)
            # store them
            pixel_dim[idx] = np.array([pixel_width, pixel_height])

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
    # save all ndarrays to a .npy files (for faster loading later)
    # Create directory to store files.
    directory = os.path.join(os.getcwd(), 'output/processed_data/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save('output/processed_data/images_both_phases_{}.npy'.format(img_rows), images)
    np.save('output/processed_data/targets_both_phases_{}.npy'.format(img_rows), targets)
    np.save('output/processed_data/pixel_dim_both_phases_{}.npy'.format(img_rows), pixel_dim)
    print('Saving to .npy files done.')


def load_train_data(model, data, img_rows=96, img_cols=96):
    """
    Loading training data & doing some additional preprocessing on it. If the indicated model is a dnn, we flatten out
    the input images. If the indicated model is a cnn, we put the channels first.
    :param model: string to indicate the type of model to prepare the data for. Either 'dnn' or 'cnn'
    :param data: Indicates which data to load (i.e data from both phases or from a specific one). Either 'ED', 'ES'
    or 'both'.
    :param img_rows: the new x-axis dimension used to resize the images
    :param img_cols: the new y-axis dimension used to resize the images
    :return: images, target features & pixel dimensions as numpy arrays.
    """
    print('#' * 30)
    print('Loading data from file. Selecting {} phase.'.format(data))
    dataname = ''

    if data == 'ED':
        dataname = 'output/processed_data/{}_phase_ED_{}.npy'
    elif data == 'ES':
        dataname = 'output/processed_data/{}_phase_ES_{}.npy'
    elif data == 'both':
        dataname = 'output/processed_data/{}_both_phases_{}.npy'

    # read in the .npy file containing the images
    images = np.load(dataname.format('images', img_rows))

    # read in the .npy file containing the target features
    targets = np.load(dataname.format('targets', img_rows))

    # read in the .npy file containing the pixel dimensions
    pixel_dim = np.load(dataname.format('pixel_dim', img_rows))

    # scale image pixel values to [0, 1]
    images = images.astype(np.float32)
    images /= 255.

    # scale target center coordinates to [-1, 1] (from 0 to 95 initially)
    targets = targets.astype(np.float32)
    targets[:, 0] = (targets[:, 0] - (img_rows / 2)) / (img_rows / 2)
    targets[:, 1] = (targets[:, 1] - (img_rows / 2)) / (img_cols / 2)

    # reshape images according to the neural network model intended to be used
    if model == 'cnn':
        print('Indicated model is a CNN, reshaping images with channels first.')
        images = images.reshape(-1, 1, img_rows, img_cols)
    elif model == 'dnn':
        print('Indicated model is a DNN, flattening out images.')
        images = images.reshape(images.shape[0], img_rows*img_rows)

    print('Loading & processing done. Pixel image values have been scaled to [0, 1],'
          'and target center coordinates to [-1, 1].')
    print('#' * 30)

    return images, targets, pixel_dim


if __name__ == '__main__':
    #create_train_data(phase=phase, img_rows=img_rows, img_cols=img_cols, verbose=0)
    create_train_data_2(img_rows=img_rows, img_cols=img_cols, verbose=0)
    X, y, pixel_dim = load_train_data(model='dnn', data='both', img_rows=img_rows, img_cols=img_cols)
    print("X.shape = {}".format(X.shape))
    print("y.shape = {}".format(y.shape))
    print("pixel_dim.shape = {}".format(pixel_dim.shape))
