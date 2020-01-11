import tensorflow as tf

if __name__=='__main__':
    tf.enable_eager_execution()

import random
import numpy as np
from tensorflow import keras
from demo.data_loaders.uci_datasets import *
from dginn.core import *
from aggregator_utils import compute_dg_per_datapoint, extract_dgs_by_ids


def step_decay(epoch, initial_lrate=0.1, drop=0.5, epochs_drop=10.0):
    '''

    Example usage:
    epochs = np.arange(1,100)
    initial_lrate = 0.1
    lr = [step_decay(e,drop=initial_lrate/0.125) for e in epochs]
    plt.plot(epochs,lr)
    '''
    import math
    #     drop = exp_decay(epoch,drop)
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    return lrate


def exp_decay(epoch, initial_lrate=0.01):
    '''
    Example usage:

    epochs = np.arange(1,100)
    lr = [exp_decay(e) for e in epochs]
    plt.plot(epochs,lr)
    '''
    k = 0.1
    t = epoch
    lrate = initial_lrate * np.exp(-k * t)
    return lrate

def build_model(input_shape=(14,), seed=49, num_layers=3, num_hidden=100, optimizer=None):
    tf.compat.v1.reset_default_graph()
    graph_level_seed = seed
    operation_level_seed = seed
    random.seed(operation_level_seed)
    np.random.seed(operation_level_seed)
    tf.compat.v1.set_random_seed(graph_level_seed)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    for i in range(num_layers):
        model.add(
            tf.keras.layers.Dense(num_hidden, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.03)))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
    if optimizer is None:
        optimizer = tf.keras.optimizers.RMSprop(0.01, decay=0.005)
    loss = tf.keras.losses.categorical_crossentropy
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def clone_model(model):
    model_orig = tf.keras.models.clone_model(model)
    model_orig.build(input_shape=model.input_shape)
    model_orig.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    return model_orig


def fit_model(model, X_train, Y_train, EPOCHS=20, batch_size=256, verbose=0):
    # from keras.callbacks import LearningRateScheduler
    # lrate = LearningRateScheduler(exp_decay)

    # The patience parameter is the amount of epochs to check for improvement
    #     early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=100)
    best_weights_filepath = './best_weights.hdf5'
    #     if os.path.exists(best_weights_filepath):
    #         os.remove(best_weights_filepath)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')
    saveBestModel = keras.callbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=0,
                                                    save_best_only=True, mode='auto')
    history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=batch_size,
                        validation_split=0.1, verbose=verbose,
                        callbacks=[early_stop])

    # reload best weights
    #     model.load_weights(best_weights_filepath)
    return history


file_path = "/Users/AdminDK/Desktop/Datasets/adult/"

model = build_model()
Xtr, Xts, ytr, yts, _, _ = get_adult(None, file_path=file_path)
X_train, _, y_train, _ = prep_data(Xtr, Xts, ytr, yts, verbose=1)
fit_model(model, X_train, y_train, verbose=1)

# SCRAP
# X_train = X
y_train = y_train

# TODO: fix this bug:
# note: weights doesn't work, because there is no aggregation, so it is missing a dimension because it still aggregates
# over data points
# Weight_Activations_Computer: also aggregates over kek points

# TODO:
# input layer is returned fully, so need to filter it somehow

#computer = Weights_Computer
computer = Activations_Computer
#computer = Weight_Activations_Computer
#computer = Gradients_Computer
dgs = compute_dg_per_datapoint(X_train, model, computer)

dg_collections_list = []
all_classes = np.unique(y_train).tolist()
for cls in all_classes:
    idx_cls = np.where(y_train == cls)[0]
    # print(idx_cls[0:10])
    dgs_cls = extract_dgs_by_ids(dgs, idx_cls)
    dg_collections_list.append(dgs_cls)


feature_indices =  dg_collections_list[0][model.layers[0]]

print("trained...")
