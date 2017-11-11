# This file is part of tdsi-deep-echo-challenge

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras import backend as K

from sklearn.model_selection import train_test_split

from data import load_train_data

batch_size = 15
num_classes = 4  # 4 target features to output
epochs = 24

# input image dimensions
img_rows, img_cols = 96, 96
input_shape = (img_rows, img_cols, 1)

K.set_image_data_format('channels_last')  # Sets the value of the data format convention.


def get_nnmodel():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


def fit_model():

    X, y = load_train_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = get_nnmodel()

    hist = model.fit(X_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(X_test, y_test))

    np.savetxt('cnn_model_loss.csv', hist.history['loss'])
    np.savetxt('cnn_model_val_loss.csv', hist.history['val_loss'])

    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # save model
    model.save('model.h5')
    print('model saved to .h5 file.')


if __name__ == '__main__':
    fit_model()

