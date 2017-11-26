# This file is part of tdsi-deep-echo-challenge

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K

from sklearn.model_selection import train_test_split

from data import load_train_data

batch_size = 15
num_classes = 4  # 4 target features to output
epochs = 1000

# input image dimensions
img_rows, img_cols = 96, 96
input_shape = (1, img_rows, img_cols)

K.set_image_data_format('channels_first')  # Sets the value of the data format convention.


def get_nnmodel():

    model = Sequential()

    model.add(Dense(100, input_dim=(img_rows * img_cols), activation='relu'))

    model.add(Dense(num_classes))

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

    model.compile(loss='mse', optimizer=sgd)

    return model


def fit_model():

    X, y = load_train_data()

    model = get_nnmodel()

    hist = model.fit(X, y, epochs=epochs, verbose=1, validation_split=0.3)

    np.savetxt('cnn_model_loss.csv', hist.history['loss'])
    np.savetxt('cnn_model_val_loss.csv', hist.history['val_loss'])

    # save model
    model.save('model.h5')
    print('model saved to .h5 file.')


if __name__ == '__main__':
    fit_model()
