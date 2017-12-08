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

#   PARAMETERS  #
datapath = 'data'  # Data path
sample_patient = 'patient0001'  # filename for plot_sample()
img_rows = 96
img_cols = 96
#################


def plot_loss(model):
    """
    Generate a matplotlib.pyplot of the metrics evolution during the model training.
    :param model: the model to select, either 'cnn' or 'dnn'
    :return: display a matplotlib.pyplot
    """
    # load data
    loss = np.loadtxt('{}_model_loss.csv'.format(model))
    acc = np.loadtxt('{}_model_acc.csv'.format(model))

    # create plot & display it
    plt.plot(loss, linewidth=2, label='Training Loss (mse)')
    plt.plot(acc, linewidth=2, label='Training accuracy')
    plt.grid()
    plt.title('Metrics evolution during epochs for {} model'.format(model.upper()))
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.show()


def plot_sample(model, sample, datapath='data/'):
    """
    Plot the predicted center & main orientation on a sample image.
    :param model: the model to use, either 'cnn' or 'dnn'
    :param sample: the sample image filename
    :param datapath: the datapath where sample is located
    :return: a matplotlib.pyplot showing predicted center & main orientation on sample image.
    """
    # load saved model
    saved_model = load_model('{}_model.h5'.format(model))

    # get sample image
    img, _, _, _ = acquisition.load_mhd_data('{d}/{p}/{p}_4CH_ES.mhd'.format(d=datapath, p=sample))
    # resize it to (img_cols, img_rows)
    img = resize(img, (img_cols, img_rows), mode='reflect', preserve_range=True)

    # scale image pixel values to [0, 1]
    img = img.astype(np.float32)
    img /= 255.

    # reshape input according to loaded model
    if model == 'dnn':
        inputimg = img.reshape(1, img_rows*img_cols)
    elif model == 'cnn':
        inputimg = img.reshape(-1, 1, 96, 96)

    prediction = saved_model.predict(inputimg, batch_size=1, verbose=1)

    # get target values (original scaling)
    row = prediction[0, 0]*(img_rows/2) + (img_rows/2)
    col = prediction[0, 1]*(img_cols/2) + (img_cols/2)
    x_v1 = prediction[0, 2]
    y_v1 = prediction[0, 3]

    print('Predicted rowCenter, colCenter = ', row, col)
    print('Predicted xOrientation, yOrientation = ', x_v1, y_v1)

    plotCenterOrientation(img, (row, col), (x_v1, y_v1))


def boxPlot():
    """
    Create a seaborn boxplot comparing DNN & CNN models on the distribution of distance between the predicted center
    & the ground truth center.
    :return: None, create seaborn boxplot.
    """
    distance = []
    label = []
    for net in ['dnn', 'cnn']:
        # load saved model
        model = load_model('{}_model.h5'.format(net))

        # get data
        X, y = load_train_data(model=net, data='both')
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        # get predictions
        net_pred = model.predict(X_test, verbose=0)

        # ground truth & predicted center coordinates are in [-1,1], scaling them back to [0, img_rows] to compute
        # the distance in pixels
        for array in [net_pred, y_test]:
            array[:, 0] = array[:, 0] * (img_rows / 2) + (img_rows / 2)
            array[:, 1] = array[:, 1] * (img_cols / 2) + (img_cols / 2)

        # compute distance between predicted center & true center and group result in a pandas dataframe
        net_distance = np.sqrt((y_test[:, 0] - net_pred[:, 0]) ** 2 + (y_test[:, 1] - net_pred[:, 1]) ** 2)
        distance = np.concatenate((distance, net_distance))
        label += [net.upper()] * net_distance.shape[0]

    df = pd.DataFrame({'Distance (pixels)': distance, 'Model used': label})

    # generate seaborn boxplot
    sns.boxplot(x='Model used', y='Distance (pixels)', data=df, orient='v')
    plt.title('Distribution of predicted distance to true center')
    plt.show()


if __name__ == '__main__':
    #plot_sample(model='cnn', sample=sample_patient, datapath=datapath)
    #plot_loss(model='cnn')
    boxPlot()
