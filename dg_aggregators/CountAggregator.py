from dg_aggregators.AbstractAggregator import AbstractAggregator
from collections import defaultdict
import numpy as np


class CountAggregator(AbstractAggregator):
    """
    An implementation of the count aggregator class
    """

    def __init__(self, dep_graphs, cls):
        """
        Keep count of every relevant node seen in all input dependency graphs
        :param dep_graphs: input dependency graphs
        :param cls: the class these dependency graphs correspond to or None for all points

        when cls == None:
        the aggregator can be used to assertain, which neurons are more regularly used then others!
        """
        self.cls = cls

        #  It will pickle if you use a named module-level function instead of a lambda.
        # source: https://stackoverflow.com/questions/2600790/multiple-levels-of-collection-defaultdict-in-python
        self.node_counts = defaultdict(lambda: defaultdict(int))
        self.n_graphs = 0

        # the challenge with the current implementation of Dependency graphs is that they use
        # keras layer as key to the dict map
        # if we create layers * neurons instances of keras layer to index node_counts
        # that will impose a significant memory overhead
        # also if we use layer: (neurons, we could potentially use the dg_intersections & unions functionality

        self.update_batch(dgs_collection=dep_graphs)

    def update_batch(self, dgs_collection):
        '''

        :param dgs_collection: dictionary: model_layer,[relevance scores for each neuron per data point]
        :return: NO return, updates self.node_counts
        '''

        for (l, relevance_matrix) in dgs_collection.items():
            neurons, counts = np.unique(relevance_matrix, return_counts=True)
            for n, c in zip(neurons, counts):
                self.node_counts[l][n] += c

        # get the value, turn them into iterator to get the first one & access the datapoint dimension
        self.n_graphs = next(iter(dgs_collection.values())).shape[0]

    # TODO at the moment the funcctio nsi meaningless
    # what needsto happen is
    # either in init or for consistency in update!
    def update(self, dep_graph):
        """
        Update neuron counts
        :param dep_graph: new dependency graph
        """

        self.n_graphs += 1

        for (l, relevance) in dep_graph.items():
            uniq_neurons, freq_count = np.unique(relevance, return_counts=True)
            for i, node in enumerate(uniq_neurons):
                self.node_counts[l][node] += freq_count[i]

    # TODO: similarity does not reflect the new usage of node_counts
    def similarity(self, dep_graph):
        """
        Compute similarity score between previously-seen dependency graphs
        and new dependency graph, based on how frequently nodes in the new
        dependency graph were seen before

        :param dep_graph: new dependency graph
        :return: similarity score between dep_graph and previously-seen graphs
        """

        # TODO: consider increasing the similarity weights of neurons closer to the output layer
        sim_score = 0
        node_count = 0

        # Sum up similarity scores for all nodes
        for (l, relevance) in dep_graph.items():
            for node in relevance:
                sim_score += self.node_counts[l][node]
                node_count += 1

        sim_score /= self.n_graphs

        return sim_score


if __name__ == '__main__':
    import tensorflow as tf
    import tensorflow.keras as keras

    tf.enable_eager_execution()
    from mnist_loader import load_mnist, build_sample_model
    import numpy as np
    from dataset_utils import filter_dataset
    from aggregator_utils import get_count_aggregators, uncertainty_pred, compute_dg
    from core import *
    import os

    # Load dataset
    X_train, y_train, X_test, y_test = load_mnist()

    # Filter out subset of classes
    selected_classes = [0, 1, 2, 3]
    X_train, y_train = filter_dataset((X_train, y_train), selected_classes)
    X_test, y_test = filter_dataset((X_test, y_test), selected_classes)

    # Create model
    input_shape = X_train.shape[1:]
    model = build_sample_model(input_shape)

    model_save_path = "../temp_models/mnist_model.h5"
    if not os.path.exists(model_save_path):
        # Train model
        model.fit(x=X_train, y=y_train, epochs=2)
        model.save_weights(model_save_path)
    else:
        model.load_weights(model_save_path)

    from dginn.aggregator_utils import extract_dgs_by_ids, compute_dg_per_datapoint

    dgs = compute_dg_per_datapoint(X_train, model, )
    count_agg = CountAggregator(dgs, None)
