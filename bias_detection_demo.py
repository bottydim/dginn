if __name__ == '__main__':
    import tensorflow as tf
    tf.enable_eager_execution()

import numpy as np
from dataset_utils import filter_dataset
from data_visualizers import visualize_samples
from aggregator_utils import compute_dg_per_datapoint, extract_dgs_by_ids
from mnist_loader import load_mnist, get_mnist_model
from core import *
from core import Activations_Computer
from dg_relevance import cf_dgs
from dg_clustering import dendrogram_clustering


def main():

    # Load dataset
    train_x, train_y, test_x, test_y = load_mnist()

    # Define biased and unbiased classes
    unbiased_classes = [0, 1, 3]
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
    n_samples = 100
    indices = np.random.choice(train_x.shape[0], n_samples, replace=False)
    train_x, train_y = train_x[indices], train_y[indices]

    # Group 4 digits into 2 classes
    train_y = train_y // 2
    test_y = test_y // 2

    # Create model
    model = get_mnist_model(train_x, train_y, 2)

    # Compute dep. graphs for every training point
    dg_collection_query = compute_dg_per_datapoint(train_x, model, Activations_Computer)

    # Compute square similarity matrix between dep. graphs
    n_samples = train_x.shape[0]
    sim_matrix = np.full((n_samples, n_samples), -1)

    for i in range(n_samples):

        print("Row: ", i)

        for j in range(i, n_samples):
            indices = [i]
            dg_1 = extract_dgs_by_ids(dg_collection_query, indices)
            indices = [j]
            dg_2 = extract_dgs_by_ids(dg_collection_query, indices)
            sim_matrix[i, j] = cf_dgs(dg_1, dg_2)

    # Symmetrise the matrix
    for i in range(n_samples):
        for j in range(n_samples):
            if sim_matrix[i, j] == -1:
                sim_matrix[i, j] = sim_matrix[j, i]


    # We need distance, not similarity
    sim_matrix = sim_matrix.astype(np.float32)
    dist_matrix = np.reciprocal(sim_matrix)
    for i in range(n_samples): dist_matrix[i, i] = 0

    dendrogram_clustering(dist_matrix, train_x)

main()