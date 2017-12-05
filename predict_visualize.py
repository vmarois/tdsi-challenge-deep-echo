# This file is part of tdsi-deep-echo-challenge

import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import load_model
from skimage.transform import resize

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deepecho import *
from data import load_train_data

data_path = 'data'  # Data path
sample_patient = 'patient0001'
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

def boxPlot():
    """
    Create a seaborn boxplot comparing DNN & CNN models on the distribution of distance between the predicted center
    & the ground truth center.
    :return: None, create seaborn boxplot.
    """
    # load saved model
    dnn_model = load_model('dnn_model.h5')
    cnn_model = load_model('cnn_model.h5')

    # get data
    X_dnn, y_dnn = load_train_data(model='dnn', data='both')
    X_cnn, y_cnn = load_train_data(model='cnn', data='both')

    _, X_test_dnn, _, y_test_dnn = train_test_split(X_dnn, y_dnn, test_size=0.4, random_state=42)
    _, X_test_cnn, _, y_test_cnn = train_test_split(X_cnn, y_cnn, test_size=0.4, random_state=42)

    # get predictions
    dnn_pred = dnn_model.predict(X_test_dnn, verbose=0)
    cnn_pred = cnn_model.predict(X_test_cnn, verbose=0)

    # scale back predicted values
    arrays = [dnn_pred, cnn_pred, y_test_dnn, y_test_cnn]
    for array in arrays:
        array[:, 0] = array[:, 0] * (img_rows / 2) + (img_rows / 2)
        array[:, 1] = array[:, 1] * (img_cols / 2) + (img_cols / 2)

    # compute distance between predicted center & true center and group result in a pandas dataframe
    dnn_distance = np.sqrt((y_test_dnn[:, 0] - dnn_pred[:, 0]) ** 2 + (y_test_dnn[:, 1] - dnn_pred[:, 1]) ** 2)
    cnn_distance = np.sqrt((y_test_cnn[:, 0] - cnn_pred[:, 0]) ** 2 + (y_test_cnn[:, 1] - cnn_pred[:, 1]) ** 2)

    distance = np.concatenate((dnn_distance, cnn_distance))
    model = np.concatenate((["DNN" for elt in dnn_distance], ["CNN" for elt in cnn_distance]))

    df = pd.DataFrame({'Distance (pixels)': distance, 'Model used': model})

    # generate seaborn boxplot
    sns.boxplot(x='Model used', y='Distance (pixels)', data=df, orient='v')
    plt.title('Distribution of predicted distance to true center')
    plt.show()


if __name__ == '__main__':
    #plot_sample(model='cnn')
    #plot_loss(model='dnn')
    boxPlot()
