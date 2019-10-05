if __name__ == '__main__':
    import tensorflow as tf

    tf.enable_eager_execution()

import os
from collections import defaultdict

import dill as pickle

from aggregator_utils import compute_dg_per_datapoint
from aggregator_utils import get_count_aggregators, extract_dgs_by_ids
from core import *
from core import Activations_Computer
from data_visualizers import visualize_samples
from dataset_utils import filter_dataset
from mnist_loader import load_mnist, get_mnist_model


def compare_points(x, aggregators):
    raise NotImplementedError()


def sort_uncertain_points(query_x, model, aggregators, show=False):
    '''

    :param aggregators:
    :param query_x: samples to compare
    :param model:
    :return:
    '''

    # Run samples through model to get predicted labels
    predictions = np.argmax(model.predict(query_x), axis=1)

    similarities = defaultdict(int)
    dg_collection_query = compute_dg_per_datapoint(query_x, model, Activations_Computer)
    for i, x_sample in enumerate(query_x):
        # print("Iteration ", i)

        # Compute dep. graph of new sample
        x_sample = np.expand_dims(x_sample, axis=0)
        indices = [i]
        dg_query = extract_dgs_by_ids(dg_collection_query, indices)

        # Obtain the sample predicted label
        y_pred = predictions[i]

        # Compute similarity of the test point to the sampled points
        similarities[i] = aggregators[y_pred].similarity(dg_query)

    # Sort points by their similarity
    sorted_keys = sorted(similarities, key=similarities.get, reverse=True)
    sorted_vals = [query_x[i] for i in sorted_keys]

    similarity_list = [similarities.get(key) for key in sorted_keys]
    # Visualise samples
    # Extract least similar 40 points
    fig_most = visualize_samples(sorted_vals[:40], similarity_list[:40], title="Most Similar to Original Class")
    # Idea: samples with lower similarity will seem stranger
    fig_least = visualize_samples(sorted_vals[::-1][:40], similarity_list[::-1][:40],
                                  title="Least Similar to Original Class")
    if show:
        plt.show(block=False)
    return fig_most, fig_least


def random_points():
    """
    Script demonstrating uncertainty functionality offered by dep. graphs.

    Sorts MNIST points by their uncertainty and prints them out.

    Idea: high uncertainty points are "strange" and seem atypical, compared to training data
    """

    # Load dataset
    train_x, train_y, test_x, test_y = load_mnist()

    # Filter out subset of classes
    selected_classes = [0, 1, 2, 3]
    train_x, train_y = filter_dataset((train_x, train_y), selected_classes)
    test_x, test_y = filter_dataset((test_x, test_y), selected_classes)

    # Create model
    # Create model
    model = get_mnist_model(train_x, train_y)

    # Select points to inspect
    n_samples = 100
    selected_points = test_x[:n_samples]

    # Create aggregators from the training samples
    aggregators = get_count_aggregators(train_x, train_y, model, n_samples)

    # Visualise points, sorted by their uncertainty
    sort_uncertain_points(selected_points, model, aggregators)


def same_class_points(cls_list, n_samples=1000):
    """
        Script demonstrating uncertainty functionality offered by dep. graphs.

        Sorts MNIST points by their uncertainty and prints them out.

        Idea: high uncertainty points are "strange" and seem atypical, compared to training data
    """

    # Load dataset
    train_x, train_y, test_x, test_y = load_mnist()
    # Create model
    model = get_mnist_model(train_x, train_y)
    # Create aggregators from the training samples
    aggregators = get_count_aggregators(train_x, train_y, model, n_samples)

    for cls in cls_list:
        # Filter out subset of classes
        selected_classes = [cls]
        # sub_train_x, train_y = filter_dataset((train_x, train_y), selected_classes)
        sub_test_x, sub_test_y = filter_dataset((test_x, test_y), selected_classes)

        # Select points to inspect
        idx = np.where(sub_test_y == cls)
        selected_points = sub_test_x[idx][:n_samples]

        # Visualise points, sorted by their uncertainty
        fig_most, fig_least = sort_uncertain_points(selected_points, model, aggregators)

        save_fig(cls, fig_most, "most", "mnist")
        save_fig(cls, fig_least, "least", "mnist")


def save_fig(cls, fig, identifier, dataset):
    plt.figure(fig.number)
    plt.savefig(os.path.join(FIG_FOLDER, "{}/{}_{}.png".format(dataset, cls, identifier)))
    with open(os.path.join(FIG_FOLDER, "{}/{}_{}.fig".format(dataset, cls, identifier)), "wb+") as f:
        pickle.dump(fig, f)


def main():
    same_class_points(list(range(10)), n_samples=10)
    # informetis(n_samples=10)


if __name__ == '__main__':
    main()
    # print(os.listdir(os.path.abspath("../")))
