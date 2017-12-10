# This file is part of tdsi-deep-echo-challenge

import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras import backend as K

from sklearn.model_selection import train_test_split

from data import load_train_data

#   PARAMETERS  #
num_classes = 4  # 4 target features to output
epochs = 35  # number of training epochs
lr_start_cnn = 0.01  # start value for decreasing learning rate (cnn model only)
lr_stop_cnn = 0.0005  # stop value for decreasing learning rate (cnn model only)
lr_dnn = 0.0001

# input image dimensions
img_rows, img_cols = 96, 96
dnn_input_shape = img_rows * img_cols
cnn_input_shape = (1, img_rows, img_cols)
##################

K.set_image_data_format('channels_first')  # Sets the value of the data format convention.


def get_dnn_model():

    model = Sequential()

    model.add(Dense(100, input_dim=dnn_input_shape, activation='relu'))

    model.add(Dense(100, activation='relu'))

    model.add(Dense(100, activation='relu'))

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(num_classes))

    sgd = SGD(lr=lr_dnn, momentum=0.99, nesterov=True)

    model.compile(loss='mse', optimizer=sgd, metrics=['acc'])

    return model


def get_cnn_model():

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=cnn_input_shape))  # should output (32, 94, 94) as 96-3+1 = 94
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # # should output (32, 47, 47)

    model.add(Conv2D(64, (2, 2)))  # should output (64, 46, 46) as 47-2+1 = 46
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # should output (64, 23, 23)

    model.add(Conv2D(128, (2, 2)))  # should output (128, 22, 22) as 23-2+1 = 22
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # should output (128, 11, 11)

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))

    sgd = SGD(lr=lr_start_cnn, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd, metrics=['acc'])

    return model


def fit_dnn_model():

    # get data : can specify if selecting one phase only or both : 'ED', 'ES, 'both'
    X, y = load_train_data(model='dnn', data='both')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # get model
    model = get_dnn_model()

    # fit model on training data
    hist = model.fit(X_train, y_train, epochs=epochs, verbose=1)

    # evaluate the model on the test data
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test mean squared error:', score[0])
    print('Test accuracy:', score[1])

    # Create directory to store metrics evolution to file.
    directory = os.path.join(os.getcwd(), 'output/metrics_evolution/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save metrics evolution
    np.savetxt('output/metrics_evolution/dnn_model_loss_{}.csv'.format(img_rows), hist.history['loss'])
    np.savetxt('output/metrics_evolution/dnn_model_acc_{}.csv'.format(img_rows), hist.history['acc'])

    # Create directory to store model to file.
    directory = os.path.join(os.getcwd(), 'output/models/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    # save model
    model.save('output/models/dnn_model_{}.h5'.format(img_rows))
    print('dnn model saved to .h5 file.')


def fit_cnn_model():

    # get data : can specify if selecting one phase only or both : 'ED', 'ES, 'both'
    X, y = load_train_data(model='cnn', data='both')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # get model
    model = get_cnn_model()

    # initialize dynamic change of learning rate : We start at the value of 'lr_start_cnn' and decrease it every epoch
    # to get to the final value of 'lr_stop_cnn'
    # initialize early stop : stop training if the monitored metric does not change for 'patience' epochs
    learning_rate = np.linspace(lr_start_cnn, lr_stop_cnn, epochs)
    change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    early_stop = EarlyStopping(monitor='loss', patience=10)

    # fit model on training data
    hist = model.fit(X_train, y_train, epochs=epochs, callbacks=[change_lr, early_stop], verbose=1)

    # evaluate the model on the test data
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test mean squared error:', score[0])
    print('Test accuracy:', score[1])

    # Create directory to store metrics evolution to file.
    directory = os.path.join(os.getcwd(), 'output/metrics_evolution/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    # save metrics evolution
    np.savetxt('output/metrics_evolution/cnn_model_loss_{}.csv'.format(img_rows), hist.history['loss'])
    np.savetxt('output/metrics_evolution/cnn_model_acc_{}.csv'.format(img_rows), hist.history['acc'])

    # Create directory to store model to file.
    directory = os.path.join(os.getcwd(), 'output/models/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    # save model
    model.save('output/models/cnn_model_{}.h5'.format(img_rows))
    print('cnn model saved to .h5 file.')


if __name__ == '__main__':
    #fit_dnn_model()
    fit_cnn_model()