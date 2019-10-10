if __name__ == '__main__':
    import tensorflow as tf
    tf.enable_eager_execution()



from mnist_loader import load_mnist, get_mnist_model
from aggregator_utils import compute_dg_per_datapoint
from aggregator_utils import get_count_aggregators, extract_dgs_by_ids
from core import *
from core import Activations_Computer
from dataset_utils import filter_dataset
import numpy as np



def compute_domain_mismatch_threshold(train_x, cls_aggregator, model):
    '''
    Compute the dep. graph similarity value used for determining whether a point is/isn't from
    the same domain
    :param train_x: set of points in the original training data, belonging to a particular class C
    :param cls_aggregator: The dep. graph aggregator for class C computed from the points in train_x
    :param model: trained model used to compute dep. graphs
    :return: the smallest dep. graph similarity of the points in train_x, compared to cls_aggregator
    '''

    # Compute dep. graphs for every training point
    dg_collection_query = compute_dg_per_datapoint(train_x, model, Activations_Computer)

    # Compute similarity of all training points compared to cls_aggregator
    n_samples = train_x.shape[0]
    similarities = np.zeros((n_samples))

    # TODO: ideally, add a vectorised form of similarity computation, in order to parallelise it

    for i in range(n_samples):

        # Compute dep. graph of next point
        indices = [i]
        dg = extract_dgs_by_ids(dg_collection_query, indices)

        # Compute similarity of point to class dep. graph
        similarities[i] = cls_aggregator.similarity(dg)

    # Compute smallest similarity
    min_sim = np.amin(similarities)

    return min_sim


def compute_mismatch_thresholds(train_x, train_y, cls_aggregators, model):
    '''
    Compute similarity thresholds for all classes in train_x, using the provided class aggregators
    :param train_x: Training data
    :param train_y: Training labels
    :param cls_aggregators: Class aggregators of dep. graphs computed from train_x
    :param model: Model trained on train_x
    :return: Returns an array of size |cls_aggregators|, with corresponding similarity thresholds
    '''

    # Get list of all classes
    all_classes = np.unique(train_y).tolist()
    all_classes.sort()

    sim_thresholds = np.zeros((len(all_classes)))

    for i, cls in enumerate(all_classes):

        # Filter out data for particular class
        cls_x, cls_y = filter_dataset((train_x, train_y), [cls])

        # Compute similarity threshold for that class
        sim_thresholds[i] = compute_domain_mismatch_threshold(cls_x, cls_aggregators[i], model)

    return sim_thresholds



def main():

    # Load dataset
    train_x, train_y, test_x, test_y = load_mnist()

    # Select a class to serve as the data from another domain
    diff_domain_cls = [2]
    diff_domain_x, diff_domain_y = filter_dataset((train_x, train_y), diff_domain_cls)

    # Filter out subset of classes to serve as train/test of data from the original domain
    selected_classes = [0, 1]
    train_x, train_y = filter_dataset((train_x, train_y), selected_classes)
    test_x, test_y = filter_dataset((test_x, test_y), selected_classes)

    # Create model
    model = get_mnist_model(train_x, train_y, 2)

    # Create aggregators from the training samples
    n_samples = 400
    aggregators = get_count_aggregators(train_x, train_y, model, n_samples)

    domain_thresholds = compute_mismatch_thresholds(train_x, train_y, aggregators, model)



    # TODO: define "uncertain" function which (in a batch):
        # takes point
        # computes label
        # computes dep. graph
        # computes sim. of dep. graph to average of cls of dep. graph
        # if sim < threshold for that class, flags that point

    # TODO: compute number of flagged points for 2s (want as much as possible)

    # TODO: compute number of flagged points for 1s (want as little as possible)

    # TODO: visualise flagged 1s and unflagged 2s
    # Expect to see: weird 2s that look like 0/1, and weird 1s that don't look ordinary

    pass





main()



