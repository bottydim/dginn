if __name__ == '__main__':
    import tensorflow as tf
    tf.enable_eager_execution()
import numpy as np
from dataset_utils import filter_dataset
from dg_aggregators.CountAggregator import CountAggregator
from dg_relevance import compute_activations_gen, \
    relevance_select, compute_weight_activations_gen, compute_grads_gen, compute_weight
from core import *


# TODO: need to move this into the dg_relevance file later on
def compute_dg(data, compute_fx):
    relevances = compute_fx(data)
    dg = relevance_select(relevances, input_layer=compute_fx.model.layers[0], threshold=0.20)
    return dg


def uncertainty_pred(dg, aggregators):
    """
    Return the class of the aggregator with the highest similarity score on the input sample
    :param x_sample: input sample
    :param aggregators: dictionary of CountAggregator objects
    :return: class of the aggregator with the highest similarity score on the given sample
    """

    max_sim, pred_label, tot_sim = -1, -1, 0

    for key in aggregators:
        aggregator = aggregators[key]
        cls = aggregator.cls
        sim = aggregator.similarity(dg)
        tot_sim += sim

        if sim > max_sim:
            max_sim = sim
            pred_label = cls

    # Return normalised similarity
    max_sim /= tot_sim

    return pred_label, max_sim


# @Dima, this is taking far too long
# I think a better appraoch could be to compute the relevances for all samples
# & then apply analysis using the relevance fx!

def get_count_aggregators(X_train, y_train, model, n_samples, mode="per_class"):
    '''

    :param X_train:
    :param y_train:
    :param model:
    :param n_samples:
    :param mode: batch is faster if we have already computed all dgs,
    maybe we want to create an inteface to pass all dgs
    :return:
    '''
    modes = ["batch", "per_class"]
    assert mode in modes
    switcher = {
        "batch": get_count_aggregators_batch,
        "per_class": get_count_aggregators_per_class,
    }
    fx_eval = switcher.get(mode, "Invalid argument")
    return fx_eval(X_train, y_train, model, n_samples)


def get_aggregators_from_collection(dg_collection_list, abstractAggregator=CountAggregator):
    aggregators = {}
    for cls in range(len(dg_collection_list)):
        dg_collection = dg_collection_list[cls]
        aggregator = CountAggregator(dg_collection, cls)
        aggregators[cls] = aggregator
    return aggregators


def get_count_aggregators_batch(X_train, y_train, model, n_samples):
    aggregators = {}

    # generate dg per data point
    dgs = compute_dg_per_datapoint(X_train, model, Activations_Computer)

    # split dgs into dg_collections, where each
    dg_collections_list = []
    all_classes = np.unique(y_train).tolist()
    for cls in all_classes:
        idx_cls = np.where(y_train == cls)[0]
        # print(idx_cls[0:10])
        dgs_cls = extract_dgs_by_ids(dgs, idx_cls)
        dg_collections_list.append(dgs_cls)

    for cls in all_classes:
        print("Aggregating class ", cls)

        # Filter out data for particular class
        cls_x, cls_y = filter_dataset((X_train, y_train), [cls])

        # Randomly extract samples from class data
        indices = np.random.choice(cls_x.shape[0], n_samples, replace=False)

        # list that contains dg_cls, which is a collection of dgs for the class data-points
        dgs_cls = dg_collections_list[cls]
        # extract the sub-sample from the dg class collection
        dgs_cls_sample = extract_dgs_by_ids(dgs_cls, indices)

        # Create aggregator from drawn samples
        aggregator = CountAggregator(dgs_cls_sample, cls)
        aggregators[cls] = aggregator
    return aggregators


def get_count_aggregators_per_class(X_train, y_train, model, n_samples):
    # Obtain all labels in the data
    all_classes = np.unique(y_train).tolist()
    all_classes.sort()

    aggregators = {}
    compute_fx = Activations_Computer(model=model, agg_data_points=True)

    for cls in all_classes:
        print("Aggregating class ", cls)

        # Filter out data for particular class
        cls_x, cls_y = filter_dataset((X_train, y_train), [cls])

        # Randomly extract samples from class data
        indices = np.random.choice(cls_x.shape[0], n_samples, replace=False)
        sub_xs, sub_ys = cls_x[indices], cls_y[indices]
        y = sub_ys[0]

        # Create count aggregator from the samples
        dgs_cls_sample = compute_dg_per_datapoint(sub_xs, model, Activations_Computer)

        # Create aggregator from drawn samples
        aggregator = CountAggregator(dgs_cls_sample, y)
        aggregators[y] = aggregator

    return aggregators


def compute_dg_per_datapoint(X_train, model, RelevanceComputer, n_layers=None):
    '''

    :param X_train: data_set
    :param model: model
    :param RelevanceComputer:
    :return:
    '''
    compute_fx = RelevanceComputer(model=model, agg_data_points=False)
    relevances = compute_fx(X_train)
    dgs = relevance_select(relevances, input_layer=compute_fx.model.layers[0], threshold=0.05)

    # TODO: this is a tmp fix. Will need to think what's the best place to put it in
    # TODO: ideally, have early stopping in dg relevance
    if n_layers is not None:
        layer_subset = model.layers[-n_layers:]
        dgs = {l: dgs[l] for l in layer_subset}

    return dgs


def extract_dgs_by_ids(dgs, idx):
    '''
    Takes dictionary: model_layer,[relevance scores for each neuron per data point]
    :param dgs: model_layer,[relevance scores for each neuron per data point]
    :param idx: the ids of the datapoints for which we need the dependency graphs
    :return:
    '''
    dg_idx = {}

    for l, relevance in dgs.items():
        values_idx = relevance[idx, :]
        # flatten relevances from shape [1,n] to [n,] if only one DG is requested
        # alternative we can convert idx from list to scalar idx =[i] => idx = i
        if len(idx) == 1:
            values_idx = values_idx.flatten()
        dg_idx[l] = values_idx
    return dg_idx

def get_number_datapoints(dg_collection):
    val = next(iter(dg_collection.values()))
    return val.shape[0]

def test_get_aggregators_from_collection():
    pass


if __name__ == '__main__':
    test_get_aggregators_from_collection()
