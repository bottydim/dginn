import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
import os, ssl


def load_cifar10():

    """
    Basic loader of CIFAR10 data
    """

    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)):
        ssl._create_default_https_context = ssl._create_unverified_context

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.0
    x_test /= 255.0

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    return x_train, y_train, x_test, y_test



def build_sample_model(input_shape, num_classes):

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model



def get_cifar_model(train_x, train_y, num_classes=10):
    input_shape = train_x.shape[1:]
    model = build_sample_model(input_shape, num_classes)
    from pathlib import Path
    path_dir = Path(os.path.dirname(os.path.abspath("__file__"))).parents[0]
    path_dir = path_dir / "dginn/temp_models"
    # TODO we want to think about the sitatuion where output is 2; yet it is trained on [0] vs [1] or [0,2] vs [1,7]
    model_save_path = path_dir / "cifar10_model_{}.h5".format(10)
    model_save_path = str(model_save_path)
    if not os.path.exists(model_save_path):
        # Train model
        model.fit(x=train_x, y=train_y, epochs=5,batch_size=1024)
        if not os.path.exists(path_dir):
            try:
                path = path_dir
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)
        model.save_weights(model_save_path)
    else:
        model.load_weights(model_save_path)

    return model


