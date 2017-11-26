# This file is part of tdsi-deep-echo-challenge

import matplotlib.pyplot as plt

from keras.models import load_model
from skimage.transform import resize

import numpy as np

from deepecho import *

data_path = 'data'  # Data path
sample_patient = 'patient0001'
img_rows = 96
img_cols = 96


def plot_loss():
    loss = np.loadtxt('cnn_model_loss.csv')
    val_loss = np.loadtxt('cnn_model_val_loss.csv')

    plt.plot(loss, linewidth=3, label='train')
    plt.plot(val_loss, linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def plot_sample():

    # load saved model
    print('-' * 30)
    print('Load model from file')
    print('-' * 30)
    model = load_model('model.h5')

    img, _, _, _ = acquisition.load_mhd_data('{d}/{p}/{p}_4CH_ES.mhd'.format(d=data_path, p=sample_patient))
    img = resize(img, (img_cols, img_rows), mode='reflect', preserve_range=True)
    input = img[np.newaxis, ..., np.newaxis]

    pred = model.predict(input, batch_size=1, verbose=1)
    row = pred[0, 0]*(img_rows/2) + (img_rows/2)
    col = pred[0, 1]*(img_cols/2) + (img_cols/2)
    x_v1 = pred[0, 1]
    y_v1 = pred[0, 3]

    plt.imshow(img, cmap='Greys')
    plt.show()

    plotCenterOrientation(img, (row, col), (x_v1, y_v1))


if __name__ == '__main__':
    plot_sample()