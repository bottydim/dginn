import numpy as np
from dataset_utils import filter_dataset
from dg_aggregators.CountAggregator import CountAggregator
from dg_relevance import compute_activations_gen, \
    relevance_select, compute_weight_activations_gen, compute_grads_gen, compute_weight


# TODO: need to move this into the dg_relevance file later on
def compute_dg(data, model):
    compute_activations_abs = compute_activations_gen(data, layer_start=1, fx_modulate=np.abs)
    relevances = compute_activations_abs(model)
    dg = relevance_select(relevances, input_layer=model.layers[0], threshold=0.9)
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


def get_count_aggregators(x_train, y_train, model, n_samples):

    # Obtain all labels in the data
    all_classes = np.unique(y_train).tolist()

    aggregators = {}

    for cls in all_classes:

        print("Aggregating class ", cls)

        # Filter out data for particular class
        cls_x, cls_y = filter_dataset((x_train, y_train), [cls])

        # Randomly extract samples from class data
        indices = np.random.choice(cls_x.shape[0], n_samples, replace=False)
        sub_xs, sub_ys = cls_x[indices], cls_y[indices]
        y = sub_ys[0]

        # Create count aggregator from the samples
        dgs = []

        # Create dep. graphs for all drawn class samples
        for (sub_x, sub_y) in zip(sub_xs, sub_ys):
            sub_x = np.expand_dims(sub_x, axis=0)
            dg = compute_dg(sub_x, model)
            dgs.append(dg)

        # Create aggregator from drawn samples
        aggregator = CountAggregator(dgs, y)
        aggregators[y] = aggregator

    return aggregators
