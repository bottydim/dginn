if __name__ == '__main__':
    import tensorflow as tf
    tf.enable_eager_execution()

from aggregator_utils import compute_dg_per_datapoint, extract_dgs_by_ids
from mnist_loader import load_mnist, get_mnist_model
from core import *
from core import Activations_Computer
from dg_relevance import cf_dgs
from dg_clustering import dendrogram_clustering
from data_loaders import load_biased_mnist_dataset

def main():

    # Load biased MNIST dataset
    train_x, train_y, _, _ = load_biased_mnist_dataset()

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

    # Plot dendrogram
    dendrogram_clustering(dist_matrix, train_x)




main()