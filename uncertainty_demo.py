
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from dg_aggregators.CountAggregator import CountAggregator
from dg_relevance import compute_activations_gen, relevance_select

def load_mnist():

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
    model.add(Conv2D(28, kernel_size=(3,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model



def filter_dataset(dataset, classes):
    """
    Only retain datapoints for specified classes
    :param dataset: Full dataset
    :param classes: Classes to filter out
    :return: Filtered dataset
    """

    (points, labels) = dataset

    mask = np.isin(labels, classes)

    filtered_points = points[mask]
    filtered_labels = labels[mask]

    return filtered_points, filtered_labels


# TODO: need to move this into the dg_relevance file later on
def compute_dg(data, model):
    compute_activations_abs = compute_activations_gen(data, fx_modulate=np.abs, layer_start=1)
    relevances = compute_activations_abs(model)
    dg = relevance_select(relevances, input_layer=model.layers[0], threshold=0.95)
    return dg




def uncertainty_pred(x_sample, aggregators):
    """
    Return the class of the aggregator with the highest similarity score on the input sample
    :param x_sample: input sample
    :param aggregators: list of AbstractAggregator objects
    :return: class of the AbstractAggregator with the highest similarity score on the given sample
    """

    sim_scores = np.array([aggregator.similarity(x_sample) for aggregator in aggregators])
    top_aggregator = aggregators[np.argmax(sim_scores)]
    predicted_class = top_aggregator.cls
    return predicted_class



def main():

    # Load dataset
    all_x_train, all_y_train, all_x_test, all_y_test = load_mnist()

    # Filter out two classes (focus on simple binary classification problem)
    classes = [0, 1]
    train_x, train_y = filter_dataset((all_x_train, all_y_train), classes)
    test_x, test_y = filter_dataset((all_x_test, all_y_test), classes)


    # Create model
    input_shape = train_x.shape[1:]
    model = build_sample_model(input_shape)

    # Train model
    model.fit(x=train_x, y=train_y, epochs=1)

    # Select set of random training samples
    n_samples = 200
    indices = np.random.choice(train_x.shape[0], n_samples, replace=False)
    x_samples = train_x[indices]
    y_samples = train_y[indices]


    # Aggregate dependency graphs of selected points
    dgs_0, dgs_1 = [], []

    for (x, y) in zip(x_samples, y_samples):

        x_sample = np.expand_dims(x, axis=0)

        dg = compute_dg(x_sample, model)

        if (y == 0):
            dgs_0.append(dg)
        else:
            dgs_1.append(dg)


    prob_dg_0 = CountAggregator(dgs_0, 0)
    prob_dg_1 = CountAggregator(dgs_1, 1)

    correct = 0
    incorrect = 0

    for (sample, label) in zip(test_x[:400], test_y[:400]):
        sample = np.expand_dims(sample, axis=0)
        dg = compute_dg(sample, model)
        sim_to_0 = prob_dg_0.similarity(dg)
        sim_to_1 = prob_dg_1.similarity(dg)
        pred_label = 0

        if sim_to_0 > sim_to_1: pred_label = 0
        else: pred_label = 1

        if pred_label == label:
            correct += 1
            print("Corr. sim. : ", sim_to_0 - sim_to_1)
        else:
            incorrect += 1
            print("Incorrect sim. : ", sim_to_0 - sim_to_1)


    acc = float(correct) / float(correct + incorrect)

    print("Dep. Graph Accuracy: ", acc * 100)


main()