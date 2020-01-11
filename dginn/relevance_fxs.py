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
    relevant = {}

    for l, relevance in omega_val.items():
        if type(input_layer) is list and l in input_layer:
            # non-aggragated data points
            if relevance.shape[0] > 1:
                relevant[l] = [range(relevance.shape[1]) for _ in range(relevance.shape[0])]
            else:
                relevant[l] = range(len(relevance))
        elif l == input_layer:
            # non-aggragated data points
            if relevance.shape[0] > 1:
                relevant[l] = np.array([range(relevance.shape[1]) for _ in range(relevance.shape[0])])
            else:
                relevant[l] = np.array(range(len(relevance)))
                # process layers without weights
        elif l.weights == []:
            # non-aggragated data points
            if relevance.shape[0] > 1:
                relevant[l] = np.array([[] for _ in range(relevance.shape[0])])
            else:
                #in the event of bugs: used to be [] vs np.array([])
                relevant[l] = np.array([])

        else:
            # non-aggragated data points
            if relevance.shape[0] > 1:
                # returns an array with the same number of neuron id for each dg
                # the array contains the neuron ids that were selected as relevant
                relevant[l] = np.apply_along_axis(select_fx_, 1, relevance)
            else:
                relevant[l] = []
                idx = select_fx_(relevance)
                relevant[l] = idx
    return relevant

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
