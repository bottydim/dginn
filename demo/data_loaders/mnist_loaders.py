from mnist_loader import load_mnist
from dataset_utils import filter_dataset
import numpy as np



def randomly_sample(x, y, n_samples):
    # Select random set of samples
    indices = np.random.choice(x.shape[0], n_samples, replace=False)
    subset_x, subset_y = x[indices], y[indices]
    return subset_x, subset_y


# TODO: add documentation
def load_biased_mnist_dataset():

    # Load MNIST dataset
    train_x, train_y, test_x, test_y = load_mnist()

    # Define biased and unbiased classes
    unbiased_classes = [0, 1, 3]
    biased_class = [2]
    all_classes = biased_class + unbiased_classes
    test_x, test_y = filter_dataset((test_x, test_y), all_classes)

    # Create dataset of unbiased classes, and class that is underrepresented
    unbiased_x, unbiased_y = filter_dataset((train_x, train_y), unbiased_classes)
    biased_x, biased_y = filter_dataset((train_x, train_y), biased_class)

    # Define number of biased and unbiased samples
    n_samples = 250
    n_biased_samples = int(n_samples * 0.1)
    n_unbiased_samples = n_samples - n_biased_samples

    # Randomly select biased and unbiased samples from their respective datasets
    biased_x, biased_y = randomly_sample(biased_x, biased_y, n_biased_samples)
    unbiased_x, unbiased_y = randomly_sample(unbiased_x, unbiased_y, n_unbiased_samples)

    # Merge and shuffle the datasets
    train_x, train_y = np.concatenate((unbiased_x, biased_x)),  np.concatenate((unbiased_y, biased_y))
    train_x, train_y = randomly_sample(train_x, train_y, train_x.shape[0])

    # Group 4 digits into 2 classes
    train_y = train_y // 2
    test_y = test_y // 2

    return train_x, train_y, test_x, test_y