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
from uncertainty_demo import sort_uncertain_points



def main():

    # Load dataset
    train_x, train_y, test_x, test_y = load_mnist()

    # Define biased and unbiased classes
    unbiased_classes = [0, 2, 3]
    biased_class = [1]
    all_classes = biased_class + unbiased_classes
    test_x, test_y = filter_dataset((test_x, test_y), all_classes)

    # Retrieve indices of biased points
    biased_points = np.where(test_y == biased_class[0])[0]

    # Create dataset of unbiased classes, and class that is underrepresented
    unbiased_x, unbiased_y = filter_dataset((train_x, train_y), unbiased_classes)
    biased_x, biased_y = filter_dataset((train_x, train_y), biased_class)
    n_biased_samples = int(biased_x.shape[0] * 0.2)
    biased_x, biased_y = biased_x[:n_biased_samples], biased_y[:n_biased_samples]
    train_x, train_y = np.concatenate((unbiased_x, biased_x)),  np.concatenate((unbiased_y, biased_y))

    # Select random set of samples
    n_samples = 800
    indices = np.random.choice(train_x.shape[0], n_samples, replace=False)
    train_x, train_y = train_x[indices], train_y[indices]

    # Group 4 digits into 2 classes
    train_y = train_y // 2
    test_y = test_y // 2

    # Create model
    model = get_mnist_model(train_x, train_y, 2)

    biased_point_idx = biased_points[40:41]
    biased_point = test_x[biased_point_idx]


    train_x_0, train_y_0 = filter_dataset((train_x, train_y), [0])

    print("")

main()

'''
Dynamic threshold computation:
    - Compute class dg
    - Compute sims of all training points
    - Compute their distribution
    - Compute outliers
    - Set as threshold


'''
