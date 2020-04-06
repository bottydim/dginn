from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from dginn.relevance_fxs import percentage_threshold
from dginn.utils import *

# this should be only in the call module, all other modules should not have it!!!
# best keep it in the main fx!

config = tf.compat.v1.ConfigProto()
# config.gpu_options.visible_device_list = str('1')
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=config)

'''
Implementation of dep. graphs, as outlined in Algorithm X in paper Y
'''


class Relevance_Computer(ABC):
    '''
    Abstract class that holds the different strategies used to compute Step I.
    '''

    def __init__(self, model, fx_modulate=lambda x: x,
                 layer_start=None,
                 agg_data_points=True,
                 agg_neurons=True,
                 verbose=False):
        '''
        :param model: TF Keras model that the dependency graphs are computed from.

        :param fx_modulate: function applied to neuron scores during neuron importance computation.
                            Example (1) : if we want "attribution" (is the feature positively or negatively contributing)
                                     then can simply use the identity function.
                            Example (2) : if we want notion of overall importance, can take absolute score value
                                          using the abs function

        :param layer_start: specifying the first layer to compute the DG from. DG is computed from the layer_start
                            up to the output layer. Can be useful for skipping initial pre-processing layers of a model

        :param agg_data_points: used with the __call__ method. Determines whether to create a single DG aggregated
                                across input data points, or to create a DG per data point

        :param agg_neurons: used with the __call__ method. Determines whether to compute importances of neurones in layer
                            L with respect to each neuron of layer (L+1), or to aggregate across all neurons of layer
                            (L+1)

        :param verbose: print intermediate computation results
        '''

        # Assign property values to object
        self.model = model
        self.fx_modulate = fx_modulate
        self.layer_start = layer_start
        self.agg_data_points = agg_data_points
        self.agg_neurons = agg_neurons
        self.verbose = verbose

    @abstractmethod
    def __call__(self, data):
        '''
        :param data: dataset to compute DGs from.
                     Numpy array. Accepted shapes: (Samples, D) or (Samples, D, T) or (Samples, W, H, C)

        :return: the DG
        '''


class Weights_Computer(Relevance_Computer):
    """
    Score: weight of the edge between the neurones
    """

    def __init__(self, model, fx_modulate=np.abs,
                 layer_start=None,
                 agg_data_points=True,
                 agg_neurons=True,
                 verbose=False):

        if verbose and not agg_data_points: print("Redundant aggregation of data points for Weights Computer")

        # Set to true since weights are the same across data points
        agg_data_points = True

        super().__init__(model, fx_modulate, layer_start, agg_data_points, agg_neurons, verbose)

    def __call__(self, data):

        model, fx_modulate, layer_start, agg_data_points, agg_neurons, verbose = self.model, self.fx_modulate, self.layer_start, self.agg_data_points, self.agg_neurons, self.verbose

        # Dictionary from layer to neuron importances in that layer
        # TODO: check whether it's that layer, or the next layer
        omega_val = {}

        for l in model.layers[layer_start:]:
            # skips layers w/o weights
            # e.g. input/pooling/concatenate/flatten
            if l.weights == []:
                vprint(l.name, verbose=verbose)
                omega_val[l] = np.array([])
                continue

            vprint("layer:{}".format(l.name), verbose=verbose)

            # 1. compute values: copy weights, and apply modulation function
            vprint("w.shape:{}".format(l.weights[0].shape), verbose=verbose)
            score_val = l.weights[0][:, :]
            score_val = fx_modulate(score_val)

            # 2 aggregate across locations
            # if convolutional - check if convolutional using the shape
            # 2.1 4D input (c.f. images)
            if len(score_val.shape) > 3:
                vprint("\tshape:{}", verbose=verbose)
                vprint("\tscore_val.shape:{}".format(score_val.shape), verbose=verbose)
                score_val = np.mean(score_val, axis=(0, 1))
            elif len(score_val.shape) > 2:
                # 3D convolutional
                vprint("\tshape:{}", verbose=verbose)
                vprint("\tscore_val.shape:{}".format(score_val.shape), verbose=verbose)
                score_val = np.mean(score_val, axis=(0))

            # 4. Global Neuron Aggregation: tokenize values across upper layer neurons
            if agg_neurons:
                score_val = np.mean(score_val, axis=-1)

            vprint("\tomega_val.shape:{}".format(score_val.shape), verbose=verbose)
            omega_val[l] = np.expand_dims(score_val, axis=0)

        return omega_val


class Activations_Computer(Relevance_Computer):
    """
    Score: activations of the layer -- output of neurons
    """

    def __init__(self, model, fx_modulate=np.abs,
                 layer_start=None,
                 agg_data_points=True,
                 agg_neurons=True,
                 verbose=False):

        super().__init__(model, fx_modulate, layer_start, agg_data_points, agg_neurons, verbose)

    def __call__(self, data):

        model, fx_modulate, layer_start, agg_data_points, agg_neurons, verbose = self.model, self.fx_modulate, self.layer_start, self.agg_data_points, self.agg_neurons, self.verbose

        omega_val = {}
        # TODO: skip concatenate+flatten layers
        # TODO: currently not aggregating accross locations of last convolutional layer of a model
        for l in model.layers[layer_start:]:
            vprint("layer:{}".format(l.name), verbose=verbose)

            # If layer is Concatenate, concatenates the input
            if type(l) is tf.keras.layers.Concatenate:
                output = tf.keras.layers.concatenate(l.input)
            else:
                output = l.input

            # Note: "output" is the output of the previous layer (l-1), which is equal to the
            # input of the first layer
            # TODO: consider renaming the "output" variable

            # TODO: faster way is to include all layers to the output array.
            # Note: this will only work for the impostor DGs, not true DGs
            model_k = tf.keras.Model(inputs=model.inputs, outputs=[output])
            score_val = model_k.predict(data)
            score_val = fx_modulate(score_val)
            vprint("layer:{}--{}".format(l.name, score_val.shape), verbose=verbose)

            # 2 aggragate across locations
            # 2.1 4D input (c.f. images)
            if len(score_val.shape) > 3:
                score_val = np.mean(score_val, axis=(1, 2))
                vprint("\t 4D shape:{}".format(score_val.shape), verbose=verbose)
            # 2.2 aggregate across 1D-input
            elif len(score_val.shape) > 2:
                score_val = np.mean(score_val, axis=(1))
                vprint("\t 3D shape:{}".format(score_val.shape), verbose=verbose)

            # 3. aggregate across datapoints
            # ToDo: Check why abs? naturally this affect previous experiments
            if agg_data_points:
                score_val = np.mean(np.abs(score_val), axis=0)

            # 4. tokenize values
            # ===redundant for activations
            omega_val[l] = score_val
            vprint("\t omega_val.shape:{}".format(score_val.shape), verbose=verbose)

        return omega_val


class Gradients_Computer(Relevance_Computer):
    """
    Score: gradient model loss wrt to the weight
    """

    def __init__(self, model, fx_modulate=lambda x: x,
                 loss=lambda x: tf.reduce_sum(x[:, :]),
                 batch_size=128,
                 layer_start=None,
                 agg_data_points=True,
                 agg_neurons=True,
                 verbose=False):

        super().__init__(model, fx_modulate, layer_start, agg_data_points, agg_neurons, verbose)

        # TODO: rename this from "loss" (not really a loss)
        # Function of data-points being differentiated
        self.loss = loss
        self.batch_size = batch_size

    def __call__(self, data, ys=None):
        model, fx_modulate, layer_start, agg_data_points, agg_neurons, verbose = self.model, self.fx_modulate, self.layer_start, self.agg_data_points, self.agg_neurons, self.verbose
        loss_ = self.loss
        batch_size = self.batch_size
        omega_val = {}

        if layer_start is None:
            layer_start = 0

        last_layer = model.layers[-1]
        omega_val[last_layer] = ...  # TODO: initialise this suitably (probably using activated output neurones)

        n_layers = len(model.layers)

        for i in range(n_layers - 1, layer_start - 1, -1):

            # TODO check for different input structures!
            # compute wrt input
            if i == 0:
                cur_layer = model.layers[0].input
                next_layer = model.layers[0]
            else:
                cur_layer = model.layers[i - 1]
                next_layer = model.layers[i]

                # skips layers w/o weights
                # e.g. input/pooling
                if cur_layer.weights == []:
                    omega_val[cur_layer] = np.array([])
                    continue

            omega_val[cur_layer] = new_gradients(data, cur_layer, next_layer, model, fx_modulate, loss_, verbose)
            vprint("\t omega_val.shape:{}".format(omega_val[cur_layer].shape), verbose=verbose)

        return omega_val

        # TODO: think about removing this, or moving to a separate function
        # Old code version:
        # for l in model.layers[layer_start:]:
        #     # skips layers w/o weights
        #     # e.g. input/pooling
        #     if l.weights == []:
        #         omega_val[l] = np.array([])
        #         continue
        #
        #     mean = gradients(data, l, model, batch_size, fx_modulate, loss_, verbose)
        #
        #     # TODO: return to previous line, once data point aggregation is fixed
        #     # omega_val[l] = mean
        #     omega_val[l] = np.expand_dims(mean, axis=0)
        #
        #     vprint("\t omega_val.shape:{}".format(mean.shape), verbose=verbose)
        # return omega_val


# TODO: decide whether to remove other "gradients" method, or rename, or...
def new_gradients(data, lower_layer, upper_layer, model, fx_modulate, loss_, verbose=0, d_wrt="activations"):
    """

    :param data:
    :param lower_layer:
    :param upper_layer: the upper layer (closer to the output)
    :param model:
    :param fx_modulate:
    :param loss_:
    :param verbose:
    :param d_wrt: differentiate with respect to activations or weighs
    :return:
    """
    # Compute gradient correctly for the last layer by changing the activation fx
    # obtain the logits of the model, instead of softmax output
    if upper_layer == model.layers[-1]:
        activation_temp = upper_layer.activation
        upper_layer.activation = tf.keras.activations.linear

    # Create submodel up to (and including) the current layer

    # special case: input
    if type(lower_layer) is tf.Tensor:
        lower_layer_output = lower_layer
    else:
        lower_layer_output = lower_layer.output

    model_k = tf.keras.Model(inputs=model.inputs, outputs=[lower_layer_output, upper_layer.output])

    # TODO: check how we handle multi-input data?!
    # TODO: fix strange layer name change : vs _
    input_names = [l.name.split(":")[0] for l in model.inputs]
    # Note: line not tested
    input_names = [l.split("_")[0] for l in input_names]

    # older version has a batched iteration with summation. Do we still need this? Yes
    # Multi-input processing
    if type(data) is dict:
        tensor = []
        # if multi-input, create a list of tensors per input
        for k in input_names:
            tensor.append(data[k].astype('float32'))
    else:
        tensor = tf.convert_to_tensor(data, dtype=tf.float32)

    with tf.GradientTape() as t:
        # special case: input
        if type(lower_layer) is tf.Tensor:
            t.watch(tensor)
        lower_layer_activations, upper_layer_activations = model_k(tensor)

    # Expected upper_layer_activations shape: [samples, y]
    # Expected lower_layer_activations shape: [samples, x]
    # Gradient output shape: [samples, y, x]

    # TODO: Do we want to differentiate with respect to weights (parameters) or activations?
    if d_wrt == "activations":
        # special case: input
        if type(lower_layer) is tf.Tensor:
            d_upper_d_lower = t.batch_jacobian(upper_layer_activations, tensor)
        else:
            d_upper_d_lower = t.batch_jacobian(upper_layer_activations, lower_layer_activations)
    elif d_wrt == "weights":
        raise NotImplementedError
        # TODO: think about the case that cur_layer input * output len(outputt) = # units in current
        # Need first dimension of target shape ((50, 100)) and source shape ((100, 100)) to match.
        # first dimension is data points
        d_upper_d_lower = t.batch_jacobian(upper_layer_activations, upper_layer.weights[0])
    elif d_wrt == "bias":
        raise NotImplementedError
        d_upper_d_lower = t.batch_jacobian(upper_layer_activations, lower_layer.weights[1])
    else:
        raise NotImplementedError

    score_val = fx_modulate(d_upper_d_lower)

    # restore output activation
    if upper_layer == model.layers[-1]:
        upper_layer.activation = activation_temp
    vprint("layer:{}--{}".format(upper_layer.name, score_val.shape), verbose=verbose)

    # TODO: does this still make sense in our new scenario, with differentiation with respect to activations?
    # # 2 aggragate across locations
    # # 2.1 4D
    # if len(score_val.shape) > 3:
    #     # (3, 3, 3, 64) -> here aggregated across data-points already
    #     # so need to average over axes (0, 1) vs (1, 2)
    #     score_val = np.mean(score_val, axis=(0, 1))
    #     vprint("\t 4D shape:{}".format(score_val.shape), verbose=verbose)
    # elif len(score_val.shape) > 2:
    #     score_val = np.mean(score_val, axis=(0))
    #     vprint("\t 3D shape:{}".format(score_val.shape), verbose=verbose)
    # # TODO: Explore how to not aggregate accross data points
    # '''
    # to include data point analysis, we can use persistant gradient_tape
    # compute point by point or use the loss
    # explore this issue for efficiency gain https://github.com/tensorflow/tensorflow/issues/4897
    # '''
    # # 4. tokenize values
    # mean = np.mean(score_val, axis=1)
    # return mean

    return score_val


def gradients(data, l, model, batch_size, fx_modulate, loss_, verbose):
    # 1. compute values
    model_k = tf.keras.Model(inputs=model.inputs, outputs=[l.output])
    # last layer of the new model
    inter_l = model_k.layers[-1]
    # NB!
    # make sure to compute gradient correctly for the last layer by changing the activation fx
    # obtain the logits of the model, instead of softmax output
    if l == model.layers[-1]:
        activation_temp = l.activation
        if l.activation != tf.keras.activations.linear:
            l.activation = tf.keras.activations.linear
    # intitialise score values
    score_val = np.zeros(inter_l.weights[0].shape)
    data_len = count_number_points(data)
    # TODO: check how we handle multi-input data?!
    # TODO: fix strange layer name change : vs _
    input_names = [l.name.split(":")[0] for l in model.inputs]
    # Note: line not tested
    input_names = [l.split("_")[0] for l in input_names]
    # split the data into batches
    for i in range(0, data_len, batch_size):
        # TODO: experiment with speed if with device(cpu:0)
        with tf.GradientTape() as t:
            # t.gradient requires tensor => convert numpy array to tensor
            # c.f. that for activations we use model.predict
            if type(data) is dict:
                tensor = []

                # if multi-input, create a list of tensors per input
                for k in input_names:
                    tensor.append(data[k][i:min(i + batch_size, data_len)].astype('float32'))
            else:
                tensor = tf.convert_to_tensor(data[i:min(i + batch_size, data_len)], dtype=tf.float32)

            # Compute gradient of loss wrt to weights for current batch
            current_loss = loss_(model_k(tensor))
            dW, _ = t.gradient(current_loss, [*inter_l.weights])

        score_val += fx_modulate(dW)
    # restore output activation
    if l == model.layers[-1]:
        l.activation = activation_temp
    vprint("layer:{}--{}".format(l.name, score_val.shape), verbose=verbose)
    # 2 aggragate across locations
    # 2.1 4D
    if len(score_val.shape) > 3:
        # (3, 3, 3, 64) -> here aggregated across data-points already
        # so need to average over axes (0, 1) vs (1, 2)
        score_val = np.mean(score_val, axis=(0, 1))
        vprint("\t 4D shape:{}".format(score_val.shape), verbose=verbose)
    elif len(score_val.shape) > 2:
        score_val = np.mean(score_val, axis=(0))
        vprint("\t 3D shape:{}".format(score_val.shape), verbose=verbose)
    # TODO: Explore how to not aggregate accross data points
    '''
    to include data point analysis, we can use persistant gradient_tape
    compute point by point or use the loss
    explore this issue for efficiency gain https://github.com/tensorflow/tensorflow/issues/4897
    '''
    # 4. tokenize values
    mean = np.mean(score_val, axis=1)
    return mean


def count_number_points(data):
    """
    :param data: input data. Could be a numpy array, or could be a dictionary of
                 {layer_name : numpy array} in case of multi-input data
    :return:
    """
    # handle multi-input data
    if type(data) is dict:
        key = list(data.keys())[0]
        data_len = data[key].shape[0]
    else:
        data_len = data.shape[0]
    return data_len


class Weight_Activations_Computer(Relevance_Computer):
    """
    Score: activation score * weight score
    """

    def __init__(self, model, fx_modulate=np.abs,
                 layer_start=None,
                 agg_data_points=True,
                 agg_neurons=True,
                 verbose=False):

        super().__init__(model, fx_modulate, layer_start, agg_data_points, agg_neurons, verbose)

    def __call__(self, data):

        model, fx_modulate, layer_start, agg_data_points, agg_neurons, verbose = self.model, self.fx_modulate, self.layer_start, self.agg_data_points, self.agg_neurons, self.verbose

        omega_val = {}
        for l in model.layers:
            # skips layers w/o weights
            # e.g. input/pooling
            if l.weights == []:
                omega_val[l] = np.array([])
                continue
            relevance_val = weight_activations_compute(data, fx_modulate, l, model, verbose)
            omega_val[l] = np.expand_dims(relevance_val, axis=0)
        return omega_val


def weight_activations_compute(data, fx_modulate, l, model, verbose):
    # 1. compute values
    # 1.1 compute activations of current layer
    model_k = tf.keras.Model(inputs=model.inputs, outputs=[l.input])
    score_val_a = model_k.predict(data)
    score_val_a = fx_modulate(score_val_a)
    # 1.2 get weights of current layer
    score_val_w = l.weights[0][:]
    score_val_w = fx_modulate(score_val_w)
    # 2 aggragate across locations
    # 2.1 Aggregate Across Activations
    # 2.1.1 aggregate across 4D
    if len(score_val_a.shape) > 3:
        score_val_a = np.mean(score_val_a, axis=(1, 2))
        vprint("\t 4D shape:{}".format(score_val_a.shape), verbose=verbose)
    # 2.1.2 aggregate across 3D(1D-input)
    elif len(score_val_a.shape) > 2:
        score_val_a = np.mean(score_val_a, axis=(1))
        vprint("\t 3D shape:{}".format(score_val_a.shape), verbose=verbose)
    # 2.2.1 aggragate across locations (WEIGHTS)
    if len(score_val_w.shape) > 3:
        vprint("\tshape:{}", verbose=verbose)
        vprint("\tscore_val.shape:{}".format(score_val_w.shape), verbose=verbose)
        score_val_w = np.mean(score_val_w, axis=(0, 1))
    elif len(score_val_w.shape) > 2:
        score_val_w = np.mean(score_val_w, axis=(0))
    # 3. aggregate activation across datapoints (Step II.)
    score_agg_a = np.mean(score_val_a, axis=0)
    # ===redundant for weights
    # 4. tokenize values (Step III.)
    # ===redundant for activations
    score_agg_w = np.mean(score_val_w, axis=-1)
    relevance_val = score_agg_a * score_agg_w
    return relevance_val


class DepGraph:
    '''
    Dependency Graph class
    '''

    def __init__(self, relevance_computer, strategy="binary"):
        self.__computer = relevance_computer
        self.__model = relevance_computer.model
        self.layer_start = relevance_computer.layer_start
        self.strategy = strategy

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model
        self.__computer.model = model

    @property
    def computer(self):
        return self.__computer

    def compute(self, X, y=None):
        '''
        Compute function, used for computing the dep. graph(s) from the input data
        :param X: input data
        :param y: one-hot encoding of the labels - could be predicted or true
        :return: dependecy graph computed from the input data
        '''

        return self.compute_variable_model(self.model, X, y)

    def compute_variable_model(self, model, X, y=None):

        data = X

        # Initialise neuron values
        filtered_neurones = {}
        # initialise using output classification neurons
        if y is None:

            # TODO: this is a tmp fix, creating a random T/F matrix for the output layer
            shape = list(model.layers[-1].output_shape)
            shape[0] = data.shape[0]
            filtered_neurones[model.layers[-1]] = np.random.choice(a=[False, True], size=shape)
        else:
            # use original shape because grad_threshold apply operations across all neurons
            true_labels = y  # vs true_labels = np.argmax(y, axis=1)
            # print("True labels shape:", true_labels.shape)
            filtered_neurones[model.layers[-1]] = true_labels.astype(bool)

        # pre-compute unfiltered neuron-neuron gradient values
        all_layer_omega_vals = self.computer(data, ys=y)

        # initialise
        current_layer_neurons = filtered_neurones[model.layers[-1]]

        # Iterate over layers, output-to-input
        # start from -2 : -1 to skip the output layer; -1 to account for 0 index
        for i in range(len(model.layers) - 2, -2, -1):

            # compute wrt input
            if i == -1:
                layer = model.layers[0].input
            else:
                layer = model.layers[i]
            # Retrieve scores of next layer
            grad_vals = all_layer_omega_vals[layer]  # Assume shape is [samples, (L+1), L]

            # Filter the important neurones for the considered layer
            current_layer_neurons = self.grad_threshold(grad_vals, current_layer_neurons, strategy=self.strategy)
            filtered_neurones[layer] = current_layer_neurons

        return filtered_neurones

    def feature_importance(self, model, X, ys=None,threshold=0.2):
        '''

        :param model:
        :param X:
        :param ys: target values (keep the signiture value the same to work with AE
        :return:
        '''
        self.strategy = "average"
        self.model = model
        if type(X) is not np.ndarray:
            X = X.numpy()
            ys = ys.numpy()
        filtered_neurons = self.compute_variable_model(self.model, X, ys,threshold=threshold)
        input_layer = model.layers[0].input
        return filtered_neurons[input_layer]

    def grad_threshold(self, omega_values, next_layer_neurones, threshold=0.2, strategy="binary"):
        '''
        Filter out important neurones, based on important neurones of next layer
        :param omega_values: Matrix of activation gradients of shape [samples, (L+1), L], where L
                                     is the number of neurons in layer L, and (L+1) is the number of neurones
                                     in layer (L+1)
        :param next_layer_neurones: Binary tensor indicating important neurones of next layer, of shape [samples, (L+1)]
        :return: Binary tensor indicating important neurones for current layer, of shape [samples, L]
        '''
        # Set scores of all entries for non-important next-layer neurones to 0
        omega_values = omega_values.numpy()
        next_layer_neurones = np.expand_dims(next_layer_neurones, axis=-1)
        masked_scores = np.multiply(next_layer_neurones, omega_values)

        # threshholding based on percentage select Top Threshold percent neurons
        select_fx_percentage = lambda relevance: percentage_threshold(relevance, threshold)
        relevant_neurons_idx = np.apply_along_axis(select_fx_percentage, axis=2, arr=masked_scores)
        thresholded_mask = np.zeros(masked_scores.shape)
        rows_idx, cols_idx, channel_idx = generate_index_arrays(thresholded_mask, relevant_neurons_idx)
        thresholded_mask[rows_idx, cols_idx, channel_idx] = masked_scores[rows_idx, cols_idx, channel_idx]

        # take the first axis since shape [samples, (L+1), L]
        if strategy == "binary":
            # TODO verify this makes sense
            curr_layer_scores = np.any(thresholded_mask, axis=1)
        elif strategy == "average":
            curr_layer_scores = np.mean(thresholded_mask, axis=1)
        return curr_layer_scores


class DGINN():
    relevant_neurons = {}
    omega_vals = {}

    def __init__(self, RelevanceComputer):
        X_train = X
        y_train = ys
        computer = Relevance_Computer

    def compute(self, xs, ys):
        model = self.model
        layer_start = self.layer_start
        compute_fx = self.compute_fx()
        dg_collections_list = compute_omega_vals(xs, ys, model, computer, agg_data_points=True)


class DGINN_1():
    relevant_neurons = {}

    def __init__(self):
        pass

    def compute(self, xs, ys):

        model = self.model
        layer_start = self.layer_start
        compute_fx = self.compute_fx()
        # variable that stores all other kwargs could be passed as a partial function @ class creation
        compute_params = self.compute_params
        verbose = self.verbose
        omega_val = {}

        # TODO NOTICE THIS IS THE GREATEST BLOCKER since we will start getting neurons compute one @ time
        # this was the problem in v0.1
        relevant_neurons = ys
        # loop over layers
        for l in model.layers[layer_start:]:
            # skips layers w/o weights
            # e.g. input/pooling
            if l.weights == []:
                omega_val[l] = np.array([])
                continue
            # Paper Steps I.,II.,III.
            # compute layer relevance values
            relevance_values = compute_fx(data, l, model, relevant_neurons, **compute_params)

            # Paper Step IV.

            # TODO: return to previous line, once data point aggregation is fixed
            # omega_val[l] = mean
            omega_val[l] = np.expand_dims(relevance_values, axis=0)

            vprint("\t omega_val.shape:{}".format(relevance_values.shape), verbose=verbose)
        return omega_val


def generate_index_arrays(thresholded_mask, index_array):
    '''

    :param thresholded_mask: array N*M*L
    :param index_array: N*M*K s.t. K<L
    :return: 3 arrays: the array to index to rows, cols, and channels respectively according to the index array


    imitate:
    for i in range(rows):
        for j in range(cols):
            print(np.array_equal(np.where(thresholded_mask[i,j]==1)[0], np.sort(result[i,j])))

    '''

    rows = thresholded_mask.shape[0]
    cols = thresholded_mask.shape[1]
    channels = index_array.shape[2]

    # produce an array such that every element is added n times to the array [6,7,8] = [6,6,7,7,8,8]
    # rows_idx = np.repeat(np.arange(rows),repeats=cols,axis=0)
    rows_idx = np.repeat(np.arange(rows), repeats=cols * channels, axis=0)
    cols_idx = np.repeat(np.broadcast_to(np.arange(cols), (rows, cols)), repeats=channels, axis=1).flatten()
    channels_idx = index_array.flatten()
    assert len(rows_idx) == len(cols_idx)
    assert len(rows_idx) == len(channels_idx)
    return rows_idx, cols_idx, channels_idx


def compute_omega_vals(X_train, y_train, model, computer, agg_data_points):
    compute_fx = computer(model=model, agg_data_points=agg_data_points)
    dg_collections_list = []
    all_classes = np.unique(y_train).tolist()
    for cls in all_classes:
        idx_cls = np.where(y_train == cls)[0]
        # print(idx_cls[0:10])
        #     dgs_cls = extract_dgs_by_ids(relevances, idx_cls)
        dgs_cls = compute_fx(X_train[idx_cls, :])
        dg_collections_list.append(dgs_cls)
    return dg_collections_list


def __main__():
    tf.compat.v1.enable_eager_execution()
    print("Core main!")
