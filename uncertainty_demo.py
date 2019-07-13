import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from dg_relevance import compute_activations_gen, relevance_select, compute_weight_activations_gen
from dg_aggregators.CountAggregator import CountAggregator
import matplotlib.pyplot as plt
import math
from collections import defaultdict


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
    compute_activations_abs = compute_activations_gen(data, layer_start=1)
    relevances = compute_activations_abs(model)
    dg = relevance_select(relevances, input_layer=model.layers[0], threshold=0.7)
    return dg



def uncertainty_pred(dg, aggregators):
    """
    Return the class of the aggregator with the highest similarity score on the input sample
    :param x_sample: input sample
    :param aggregators: dictionary of CountAggregator objects
    :return: class of the aggregator with the highest similarity score on the given sample
    """

    max_sim, pred_label = -1, -1

    for key in aggregators:
        aggregator = aggregators[key]
        cls = aggregator.cls
        sim = aggregator.similarity(dg)

        if sim > max_sim:
            max_sim = sim
            pred_label = cls

    return pred_label, max_sim



def visualize_samples(samples):

    cols = 4
    rows = math.ceil(len(samples)/cols)
    fig = plt.figure(figsize=(8, 8))

    for i in range(1, len(samples)+1):

        # Remove channel dimension
        sample = samples[i-1][:, :, 0]

        # Plot image
        fig.add_subplot(rows, cols, i)
        plt.imshow(sample, cmap='gray')

    plt.show()




def get_count_aggregators(x_train, y_train, model, n_samples):

    # Obtain all labels in the data
    all_classes = np.unique(y_train).tolist()

    aggregators = {}

    for cls in all_classes:

        # Filter out data for particular class
        cls_x, cls_y = filter_dataset((x_train, y_train), [cls])

        # Randomly extract samples from class data
        indices = np.random.choice(cls_x.shape[0], n_samples, replace=False)
        sub_xs, sub_ys = cls_x[indices], cls_y[indices]
        y = sub_ys[0]

        # Create count aggregator from the samples
        dgs = []

        # Create dep. graphs for all drawn class samples
        for (sub_x, sub_ys) in zip(sub_xs, sub_ys):
            sub_x = np.expand_dims(sub_x, axis=0)
            dg = compute_dg(sub_x, model)
            dgs.append(dg)

        # Create aggregator from drawn samples
        aggregator = CountAggregator(dgs, y)
        aggregators[y] = aggregator

    return aggregators



def sort_uncertain_points(x_samples, model, n_samples=100):


    # Run samples through model to get predicted labels
    predictions = np.argmax(model.predict(x_samples), axis=1)

    aggregators = get_count_aggregators(x_samples, predictions, n_samples)

    similarities = {}

    for i, x_sample in enumerate(x_samples):

        print("Iteration ", i)

        # Compute dep. graph of new sample
        x_sample = np.expand_dims(x_sample, axis=0)
        dg_query = compute_dg(x_sample, model)

        # Obtain the sample predicted label
        y_pred = predictions[i]

        # Compute similarity of the test point to the sampled points
        similarities[i] = aggregators[y_pred].similarity(dg_query)


    # Sort points by their similarity
    sorted_keys = sorted(similarities, key=similarities.get)
    sorted_vals = [x_samples[i] for i in sorted_keys]

    # Extract least similar 30 points
    sorted_vals = sorted_vals[:40]

    visualize_samples(sorted_vals)






def run_uncertainty_demo():
    """
    Script demonstrating uncertainty functionality offered by dep. graphs.

    Sorts MNIST points by their uncertainty and prints them out.

    Idea: high uncertainty points are "strange" and seem atypical, compared to training data
    """

    # Load dataset
    train_x, train_y, test_x, test_y = load_mnist()

    # Split datasets by label into sub-datasets
    sub_datasets = {}

    for label in range(10):
        ds, _ = filter_dataset((train_x, train_y), [label])
        sub_datasets[label] = ds


    # Create model
    input_shape = train_x.shape[1:]
    model = build_sample_model(input_shape)

    # Train model
    #model.fit(x=train_x, y=train_y, epochs=4)


    # Select points to inspect
    selected_points = test_x[:100]

    # Visualise points, sorted by their uncertainty
    sort_uncertain_points(selected_points, sub_datasets, model, n_samples=100)




def mislabelled_data_demo():

    # Load dataset
    train_x, train_y, test_x, test_y = load_mnist()

    # Create model
    input_shape = train_x.shape[1:]
    model = build_sample_model(input_shape)

    # Train model
    model.fit(x=train_x, y=train_y, epochs=2)

    # Get sample aggregators
    aggregators = get_count_aggregators(train_x, train_y, model, 60)

    # Obtain subset of incorrectly-labelled training points
    preds = np.argmax(model.predict(train_x), axis=1)
    incorrects = np.nonzero(preds != train_y)[0]
    incorrect_samples, incorrect_labels = train_x[incorrects], train_y[incorrects]
    incorrect_samples, incorrect_labels = incorrect_samples[:100], incorrect_labels[:100]

    # Select random set of samples
    n_mislabelled = 100
    indices = np.random.choice(train_x.shape[0], n_mislabelled, replace=False)
    random_x, random_y = train_x[indices], train_y[indices]

    combined_x, combined_y = np.concatenate((incorrect_samples, random_x), axis=0), \
                             np.concatenate((incorrect_labels, random_y), axis=0)

    print("No samples: ", combined_x.shape[0])

    sim_scores = {}

    for i, sample in enumerate(combined_x):

        print("Iteration ", i)

        sample = np.expand_dims(sample, axis=0)
        dg = compute_dg(sample, model)
        pred_label, sim_score = uncertainty_pred(dg, aggregators)
        sim_scores[i] = sim_score


    sorted_keys = sorted(sim_scores, key=sim_scores.get)

    mislabelled = sum([1 for key in sorted_keys[:100] if key > 100])
    print("Mislabelled: ", mislabelled)

    return sorted_keys







def main():

    print(mislabelled_data_demo())

    # run_uncertainty_demo()



main()