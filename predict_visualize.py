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
img_rows = 128
img_cols = 128
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


def plot_sample(model, sample, datapath='data/', phase='ED'):
    """
    Plot the predicted center & main orientation on a sample image.
    :param model: the model to use, either 'cnn' or 'dnn'
    :param sample: the sample image filename
    :param datapath: the datapath where sample is located
    :param phase: indicates which phase to select, either 'ED' or 'ES'
    :return: a matplotlib.pyplot showing predicted center & main orientation on sample image.
    """
    # load saved model
    saved_model = load_model('{}_model.h5'.format(model))

    # get sample image
    img, _, _, _ = acquisition.load_mhd_data('{d}/{pa}/{pa}_4CH_{ph}.mhd'.format(d=datapath, pa=sample, ph=phase))

    # get associated mask
    img_mask, _, _, _ = acquisition.load_mhd_data('{d}/{pa}/{pa}_4CH_{ph}_gt.mhd'.format(d=datapath, pa=sample, ph=phase))

    # resize it to (img_cols, img_rows)
    img = resize(img, (img_cols, img_rows), mode='reflect', preserve_range=True)
    img_mask = resize(img_mask, (img_cols, img_rows), mode='reflect', preserve_range=True)

    # get roi & find 'true' center
    img_mask = getRoi(img_mask, 1)
    true_row, true_col = findCenter(img_mask)

    # scale image pixel values to [0, 1]
    img = img.astype(np.float32)
    img /= 255.

    # reshape input according to loaded model
    if model == 'dnn':
        inputimg = img.reshape(1, img_rows*img_cols)
    elif model == 'cnn':
        inputimg = img.reshape(-1, 1, img_rows, img_cols)

    # get prediction on input image
    prediction = saved_model.predict(inputimg, batch_size=1, verbose=1)

    # get target values (original scaling)
    pred_row = prediction[0, 0]*(img_rows/2) + (img_rows/2)
    pred_col = prediction[0, 1]*(img_cols/2) + (img_cols/2)
    x_v1 = prediction[0, 2]
    y_v1 = prediction[0, 3]

    # print some info
    print('True rowCenter, colCenter = ', true_row, true_col)
    print('Predicted rowCenter, colCenter = ', int(pred_row), int(pred_col))
    print('Predicted xOrientation, yOrientation = ', x_v1, y_v1)

    scale = 35
    # plot resized image
    plt.imshow(img, cmap='Greys_r')
    # plot orientation line passing through predicted center
    plt.plot([pred_col - x_v1 * scale, pred_col + x_v1 * scale],
             [pred_row - y_v1 * scale, pred_row + y_v1 * scale],
             color='white')

    fig = plt.gcf()
    ax = fig.gca()

    # plot predicted center
    pred_center = plt.Circle((pred_col, pred_row), 1, color='red')
    ax.add_artist(pred_center)
    ax.add_artist(pred_center)
    # plot true center
    true_center = plt.Circle((true_col, true_row), 1, color='black')
    ax.add_artist(true_center)

    plt.axis('equal')
    plt.title('True & predicted center +  predicted orientation. Model = {} '.format(model.upper()))
    plt.show()


def boxPlot():
    """
    Create a seaborn boxplot comparing DNN & CNN models on the distribution of distance between the predicted center
    & the ground truth center.
    :return: None, create a seaborn boxplot.
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
    plot_sample(model='dnn', sample=sample_patient, datapath=datapath, phase='ES')
    #plot_loss(model='dnn')
    #boxPlot()
