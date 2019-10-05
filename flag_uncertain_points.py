import numpy as np
from mnist_loader import load_mnist, build_mnist_model

from dataset_utils import filter_dataset
from data_visualizers import visualize_samples
from aggregator_utils import get_count_aggregators, compute_dg
from dg_aggregators.CountAggregator import CountAggregator





def main():

    # Load dataset
    train_x, train_y, test_x, test_y = load_mnist()

    # Define biased and unbiased classes
    unbiased_classes = [3,4,5]
    biased_class = [2]
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
    train_y = train_y // 4
    test_y = test_y // 4

    # Create model
    input_shape = train_x.shape[1:]
    model = build_mnist_model(input_shape, len(all_classes))

    # Train model
    model.fit(x=train_x, y=train_y, epochs=1)


    biased_point_idx = biased_points[10]
    biased_point = test_x[biased_point_idx]
