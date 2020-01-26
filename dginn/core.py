from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from dginn.utils import *

# this should be only in the call module, all other modules should not have it!!!
# best keep it in the main fx!

config = tf.ConfigProto()
# config.gpu_options.visible_device_list = str('1')
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)


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
                                     then can simply use identify function.
                            Example (2) : if we want notion of overall importance, can take absolute score value
                                          using the abs function

        :param layer_start: specifying the first layer to compute the DG from. DG is computed from the layer_start
                            up to the output layer. Can be useful for skipping initial pre-processing layers of a model

        :param agg_data_points: used with the __call__ method. Determines whether to create a single DG aggregated
                                accross input data points, or to create a DG per data point

        :param agg_neurons: used with the __call__method. Determines whether to compute importances of neurones in layer
                            L with respect to each neuron of layer (L+1), or to aggregate accross all neurons of layer
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

            # 2 aggragate across locations
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
            omega_val[l] = np.expand_dims(score_val,axis=0)

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

            # If layer is Concatenate, concetenates the input
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

    def __call__(self, data):
        model, fx_modulate, layer_start, agg_data_points, agg_neurons, verbose = self.model, self.fx_modulate, self.layer_start, self.agg_data_points, self.agg_neurons, self.verbose
        loss_ = self.loss
        batch_size = self.batch_size
        omega_val = {}

        for l in model.layers[layer_start:]:
            # skips layers w/o weights
            # e.g. input/pooling
            if l.weights == []:
                omega_val[l] = np.array([])
                continue

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

            data_len = self.count_number_points(data)

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

            # restor output activation
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
            # omega_val[l] = mean

            # TODO: return to previous line, once data point aggregation is fixed
            omega_val[l] = np.expand_dims(mean, axis=0)

            vprint("\t omega_val.shape:{}".format(mean.shape), verbose=verbose)
        return omega_val


    def count_number_points(self, data):
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

            # 3. aggregate activation across datapoints
            score_agg_a = np.mean(score_val_a, axis=0)

            # ===redundant for weights
            # 4. tokenize values
            # ===redundant for activations
            score_agg_w = np.mean(score_val_w, axis=-1)
            omega_val[l] = score_agg_a * score_agg_w
            omega_val[l] = np.expand_dims(omega_val[l] , axis=0)
        return omega_val


def __main__():
    tf.enable_eager_execution()
    print("Core main!")
