import os

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer


def load_mnist():
    """
    Basic loader of MNIST data
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.0
    x_test /= 255.0

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    return x_train, y_train, x_test, y_test


def build_sample_model(input_shape):
    """
    Builds a simple model for processing the MNIST dataset
    :param input_shape: shape of images (height, width, channels)
    :return: constructed model
    """

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(28, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(64, activation=tf.nn.relu))
    model.add(Dense(32, activation=tf.nn.relu))
    model.add(Dense(4, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def get_mnist_model(train_x, train_y):
    input_shape = train_x.shape[1:]
    model = build_sample_model(input_shape)
    from pathlib import Path

    path_dir = Path(os.path.dirname(os.path.abspath("__file__"))).parents[0]
    model_save_path = path_dir / "dginn/temp_models/mnist_model.h5"
    model_save_path = str(model_save_path)
    if not os.path.exists(model_save_path):
        # Train model
        model.fit(x=train_x, y=train_y, epochs=2)
        model.save_weights(model_save_path)
    else:
        model.load_weights(model_save_path)

    return model
