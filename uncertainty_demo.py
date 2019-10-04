from loadData import load_informetis

if __name__ == '__main__':
    import tensorflow as tf

    tf.enable_eager_execution()

from collections import defaultdict
from mnist_loader import load_mnist, get_mnist_model
from dataset_utils import filter_dataset
from data_visualizers import visualize_samples
from aggregator_utils import get_count_aggregators, compute_dg_per_datapoint, extract_dgs_by_ids
from core import *


def compare_points(x, aggregators):
    raise NotImplementedError()


def sort_uncertain_points(query_x, model, aggregators):
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
        fig_most
        plt.savefig("../../../figures/mnist_{}_most.png".format(cls))
        fig_least
        plt.savefig("../../../figures/mnist_{}_least.png".format(cls))


def informetis(cls=0):
    cls_datasets, model = load_informetis()

    from aggregator_utils import compute_dg_per_datapoint
    from dginn.core import Activations_Computer
    # dist lists
    dg_collection_list = []
    for i in range(len(cls_datasets)):
        print("Dataset #{}".format(i))
        dg_collection = compute_dg_per_datapoint(cls_datasets[i], model, Activations_Computer)
        dg_collection_list.append(dg_collection)

    from dginn.aggregator_utils import get_aggregators_from_collection, get_number_datapoints

    aggregators = get_aggregators_from_collection(dg_collection_list)
    dg_collection_query = dg_collection_list[cls]
    similarities = {}
    num_datapoints = get_number_datapoints(dg_collection_query)
    for i in range(num_datapoints):
        dg_query = extract_dgs_by_ids(dg_collection_query, [i])

        # Compute similarity of the test point to the sampled points
        similarities[i] = aggregators[cls].similarity(dg_query)

    # Sort points by their similarity
    sorted_keys = sorted(similarities, key=similarities.get, reverse=True)

    from data_visualizers import visualize_samples_informetis
    samples = cls_datasets[cls]["aggPower"][sorted_keys]
    visualize_samples_informetis(samples, similarities, title="Most Similar")


def informetis_prototypical(dataset, dg_collection_query, aggregator):
    from dginn.aggregator_utils import get_aggregators_from_collection, get_number_datapoints, extract_dgs_by_ids

    similarities = {}
    num_datapoints = get_number_datapoints(dg_collection_query)

    for i in range(num_datapoints):
        dg_query = extract_dgs_by_ids(dg_collection_query, [i])

        # Compute similarity of the test point to the sampled points
        similarities[i] = aggregator.similarity(dg_query)

    # Sort points by their similarity
    sorted_keys = sorted(similarities, key=similarities.get, reverse=True)

    from data_visualizers import visualize_samples_informetis
    samples = dataset["aggPower"][sorted_keys]
    similarity_list = [similarities.get(key) for key in sorted_keys]

    similarity_list = [similarities.get(key) for key in sorted_keys]
    lim_samples = 10
    visualize_samples_informetis(samples[:lim_samples, ...], similarity_list[:lim_samples], figsize=(20, 15))

    samples_rev = samples[::-1]
    visualize_samples_informetis(samples_rev[:lim_samples, ...], similarity_list[::-1][:lim_samples])
    return samples, similarity_list


def main():
    same_class_points(list(range(10)), n_samples=1000)
    # informetis()


if __name__ == '__main__':
    main()
