import numpy as np
import math
##########GENERATE DG##########
# tokenize


# functions, which handle the neuron relevance selection
def percentage_threshold(relevance, t):
    r = np.ravel(relevance)
    # sort args based on val, reverse, take %
    return list(reversed(np.argsort(r)))[:math.ceil(len(r) * t)]


def select_random(relevance, threshold):
    size = len(relevance)
    return np.random.choice(size, threshold)


def relevance_select_(omega_val, input_layer, select_fx_):
    '''

    :param omega_val: dict of layer: omega_values
    :param input_layer:
    :param select_fx_:
    :return:
    '''
    relevant = {}

    for l, relevance in omega_val.items():
        relevant[l] = procesess_relevance_values(relevance, l, select_fx_, input_layer)
    return relevant


def procesess_relevance_values(relevance, l, select_fx_, input_layer):
    """

    :param relevance: 2D-array data-points x neurons (to be filtered) should be lower layer
    :param l: layer corresponding to the neurons
    :param select_fx_: function to apply the relevance selection
    :param input_layer: pass the input layer or list of input layers, if l is such
    :return:
    """
    if type(input_layer) is list and l in input_layer:
        # non-aggragated data points
        if relevance.shape[0] > 1:
            return [range(relevance.shape[1]) for _ in range(relevance.shape[0])]
        else:
            return range(len(relevance))
    elif l == input_layer:
        # non-aggragated data points
        if relevance.shape[0] > 1:
            return  np.array([range(relevance.shape[1]) for _ in range(relevance.shape[0])])
        else:
            return np.array(range(len(relevance)))
            # process layers without weights
    elif l.weights == []:
        # non-aggragated data points
        if relevance.shape[0] > 1:
            return  np.array([[] for _ in range(relevance.shape[0])])
        else:
            # in the event of bugs: used to be [] vs np.array([])
            return np.array([])

    else:
        # non-aggragated data points
        if relevance.shape[0] > 1:
            # returns an array with the same number of neuron id for each dg
            # the array contains the neuron ids that were selected as relevant
            return np.apply_along_axis(select_fx_, 1, relevance)
        else:
            return []
            idx = select_fx_(relevance)
            return idx


# 3 functions below take
# omega value: the compute relevances from the previous step
# input_layer: KEEP ALL neurons LAYER - either list or single layer
# threshold: varies depending on relevance_select_fx
def relevance_select_random(omega_val, input_layer, threshold):
    select_fx_random = lambda relevance: select_random(relevance, threshold)
    return relevance_select_(omega_val, input_layer, select_fx_random)


def relevance_select(omega_val, input_layer, threshold=0.5):
    select_fx_percentage = lambda relevance: percentage_threshold(relevance, threshold)
    return relevance_select_(omega_val, input_layer, select_fx_percentage)



def relevance_select_mean(omega_val, input_layer):
    def select_fx_mean(relevance):
        threshold = np.mean(relevance)
        idx = np.where(relevance > threshold)[0]
        return idx

    return relevance_select_(omega_val, input_layer, select_fx_mean)
