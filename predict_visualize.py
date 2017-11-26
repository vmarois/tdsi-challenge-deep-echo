# This file is part of tdsi-deep-echo-challenge

import matplotlib.pyplot as plt

from keras.models import load_model
from skimage.transform import resize

import numpy as np

from deepecho import *

data_path = 'data'  # Data path
sample_patient = 'patient0200'
img_rows = 96
img_cols = 96


def plot_loss(model):
    loss = np.loadtxt('{}_model_loss.csv'.format(model))
    acc = np.loadtxt('{}_model_acc.csv'.format(model))

    plt.plot(loss, linewidth=3, label='loss')
    plt.plot(acc, linewidth=3, label='training accuracy')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('metrics')
    plt.show()


def plot_sample(model):

    # load saved model
    print('-' * 30)
    print('Load model from file')
    print('-' * 30)
    saved_model = load_model('{}_model.h5'.format(model))

    # get sample image
    img, _, _, _ = acquisition.load_mhd_data('{d}/{p}/{p}_4CH_ES.mhd'.format(d=data_path, p=sample_patient))
    img = resize(img, (img_cols, img_rows), mode='reflect', preserve_range=True)

    # scale image pixel values to [0, 1]
    img = img.astype(np.float32)
    img /= 255.

    # reshape input according to loaded model
    if model == 'dnn':
        input = img.reshape(1, img_rows*img_cols)
    elif model == 'cnn':
        input = img.reshape(-1, 1, 96, 96)

    # just to verify, print input shape
    print(input.shape)

    pred = saved_model.predict(input, batch_size=1, verbose=1)

    # get target values (original scaling)
    row = pred[0, 0]*(img_rows/2) + (img_rows/2)
    col = pred[0, 1]*(img_cols/2) + (img_cols/2)
    x_v1 = pred[0, 2]
    y_v1 = pred[0, 3]

    print('rowCenter, colCenter = ', row, col)
    print('xOrientation, yOrientation = ', x_v1, y_v1)

    plotCenterOrientation(img, (row, col), (x_v1, y_v1))


if __name__ == '__main__':
    plot_sample(model='dnn')
    #plot_loss(model='dnn')
