if __name__ == '__main__':
    import tensorflow as tf

    tf.enable_eager_execution()

import os
from collections import defaultdict

from aggregator_utils import compute_dg_per_datapoint
from aggregator_utils import get_count_aggregators, extract_dgs_by_ids
from core import *
from core import Activations_Computer
from data_visualizers import visualize_samples
from dataset_utils import filter_dataset
from mnist_loader import load_mnist, get_mnist_model

def main():

    # Load dataset
    train_x, train_y, test_x, test_y = load_mnist()

    # Define biased and unbiased classes
    unbiased_classes = [1, 2, 6]
    biased_class = [8]
    all_classes = biased_class + unbiased_classes
    test_x, test_y = filter_dataset((test_x, test_y), all_classes)

    # Retrieve indices of biased points
    biased_points = np.where(test_y == biased_class[0])[0]

    # Create dataset of unbiased classes, and class that is underrepresented
    unbiased_x, unbiased_y = filter_dataset((train_x, train_y), unbiased_classes)
    biased_x, biased_y = filter_dataset((train_x, train_y), biased_class)
    n_biased_samples = int(biased_x.shape[0] * 0.1)
    biased_x, biased_y = biased_x[:n_biased_samples], biased_y[:n_biased_samples]
    train_x, train_y = np.concatenate((unbiased_x, biased_x)),  np.concatenate((unbiased_y, biased_y))

    # Select random set of samples
    n_samples = 800
    indices = np.random.choice(train_x.shape[0], n_samples, replace=False)
    train_x, train_y = train_x[indices], train_y[indices]

    # Group 4 digits into 2 classes
    train_y = train_y // 6
    test_y = test_y // 6

    # Create model
    model = get_mnist_model(train_x, train_y, 2)

    #### in case of crash try chaning to biased_points[30:31]
    biased_point_idx = biased_points[42]
    biased_point = test_x[biased_point_idx]



    train_x_0, train_y_0 = filter_dataset((train_x, train_y), [1])

    biased_point_comparison(biased_point,train_x_0,model)



def biased_point_comparison(biased_point, biased_x, model):
    similarities = defaultdict(int)
    # expand dims
    # n_biased = biased_points.shape[0]
    biased_point = np.expand_dims(biased_point, axis=0)
    aggregator = get_count_aggregators(biased_point, np.zeros((1)), model, n_samples=1)[0]
    dg_collection_query = compute_dg_per_datapoint(biased_x, model, Activations_Computer)
    for i, x_sample in enumerate(biased_x):
        # print("Iteration ", i)

        # Compute dep. graph of new sample
        x_sample = np.expand_dims(x_sample, axis=0)
        indices = [i]
        dg_query = extract_dgs_by_ids(dg_collection_query, indices)

        # Compute similarity of the test point to the sampled points
        similarities[i] = aggregator.similarity(dg_query)


    # Sort points by their similarity
    sorted_keys = sorted(similarities, key=similarities.get, reverse=True)
    sorted_vals = [biased_x[i] for i in sorted_keys]
    similarity_list = [similarities.get(key) for key in sorted_keys]

    # Visualise test sample
    fig_query = visualize_samples(biased_point, similarity_list[0:1], title="Original sample")

    # Visualise samples
    # Extract least similar 40 points
    fig_most = visualize_samples(sorted_vals[:40], similarity_list[:40], title="Most Similar to Original Class")
    plt.show(block=False)
    # Idea: samples with lower similarity will seem stranger
    fig_least = visualize_samples(sorted_vals[::-1][:40], similarity_list[::-1][:40],
                                  title="Least Similar to Original Class")
    plt.show(block=False)

if __name__ == '__main__':
    main()

'''
Dynamic threshold computation:
    - Compute class dg
    - Compute sims of all training points
    - Compute their distribution
    - Compute outliers
    - Set as threshold
'''
