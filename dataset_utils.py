import numpy as np


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
