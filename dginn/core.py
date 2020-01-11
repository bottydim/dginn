from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from utils import *

# this should be only in the call module, all other modules should not have it!!!
# best keep it in the main fx!

config = tf.ConfigProto()
# config.gpu_options.visible_device_list = str('1')
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)


class Relevance_Computer(ABC):
    '''
    Abstract class that holds the different strategies used to compute Step I.
    '''

    def __init__(self, model, fx_modulate=lambda x: x,
                 layer_start=None,
                 agg_data_points=True,
                 local=False,
                 verbose=False):
        self.model = model
        self.fx_modulate = fx_modulate
        self.layer_start = layer_start
        self.agg_data_points = agg_data_points
        self.local = local
        self.verbose = verbose

    @abstractmethod
    def __call__(self, data):
        '''

        :param data: dataset to compute neuron relevances for
        :return: th
        '''


class Weights_Computer(Relevance_Computer):
    def __init__(self, model,
                 fx_modulate=np.abs,
                 layer_start=None,
                 agg_data_points=True,
                 agg_neurons=False,
                 verbose=False):
        super().__init__(model, fx_modulate, layer_start, agg_data_points, agg_neurons, verbose)
        if agg_data_points:
            print("this property is reduntant for Weights Computer")
        if layer_start is not None:
            print("this property is reduntant for Weights Computer")

    def __call__(self, data):

        model, fx_modulate, layer_start, agg_data_points, agg_neurons, verbose = self.model, self.fx_modulate, self.layer_start, self.agg_data_points, self.local, self.verbose
        omega_val = {}
        for l in model.layers:
            # skips layers w/o weights
            # e.g. input/pooling/concatenate/flatten
            if l.weights == []:
                vprint(l.name, verbose=verbose)
                omega_val[l] = np.array([])
                continue
            vprint("layer:{}".format(l.name), verbose=verbose)
            # 1. compute values
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
            # 3. aggregate across datapoints
            # ===redundant for weights
            # 4. Global Neuron Aggregation: tokenize values across upper layer neurons
            if not agg_neurons:
                score_agg = np.mean(score_val, axis=-1)
            else:
                score_agg = score_val
            vprint("\tomega_val.shape:{}".format(score_agg.shape), verbose=verbose)
            omega_val[l] = score_agg

        return omega_val


class Activations_Computer(Relevance_Computer):
    def __init__(self, model, fx_modulate=np.abs,
                 layer_start=None,
                 agg_data_points=True,
                 local=False,
                 verbose=False):
        super().__init__(model, fx_modulate, layer_start, agg_data_points, local, verbose)

    def __call__(self, data):

        model, fx_modulate, layer_start, agg_data_points, local, verbose = self.model, self.fx_modulate, self.layer_start, self.agg_data_points, self.local, self.verbose
        omega_val = {}
        # it is useless to compute concatenate & flatten layers!
        # also it could be useful to solve the transition between the last convo & first dense!
        for l in model.layers[layer_start:]:
            vprint("layer:{}".format(l.name), verbose=verbose)
            # concetenates the input for concatenate layers
            if type(l) is tf.keras.layers.Concatenate:
                output = tf.keras.layers.concatenate(l.input)
            else:
                output = l.input
            # faster way is to include all layers to the output array!
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
            # ? Why abs? naturally this affect previous experiments
            if agg_data_points:
                score_val = np.mean(np.abs(score_val), axis=0)

            # 4. tokenize values
            # ===redundant for activations
            omega_val[l] = score_val
            vprint("\t omega_val.shape:{}".format(score_val.shape), verbose=verbose)

        return omega_val


# TODO
# remaining functions

class Gradients_Computer(Relevance_Computer):
    def __init__(self, model, fx_modulate=lambda x: x,
                 loss=lambda x: tf.reduce_sum(x[:, :]),
                 batch_size=128,
                 layer_start=None,
                 agg_data_points=True,
                 local=False,
                 verbose=False):
        super().__init__(model, fx_modulate, layer_start, agg_data_points, local, verbose)
        self.loss = loss
        self.batch_size = batch_size

    def __call__(self, data):
        model, fx_modulate, layer_start, agg_data_points, local, verbose = self.model, self.fx_modulate, self.layer_start, self.agg_data_points, self.local, self.verbose
        loss_ = self.loss
        batch_size = self.batch_size
        omega_val = {}
        for l in model.layers:
            # skips layers w/o weights
            # e.g. input/pooling
            if l.weights == []:
                omega_val[l] = np.array([])
                continue
                # 1. compute values
            #
            model_k = tf.keras.Model(inputs=model.inputs, outputs=[l.output])
            # last layer of the new model
            inter_l = model_k.layers[-1]

            # NB!
            # make sure to compute gradient correctly for the last layer by chaning the activation fx
            if l == model.layers[-1]:
                activation_temp = l.activation
                if l.activation != tf.keras.activations.linear:
                    l.activation = tf.keras.activations.linear

            score_val = np.zeros(inter_l.weights[0].shape)

            # handle multi-input data

            if type(data) is dict:
                key = list(data.keys())[0]
                data_len = data[key].shape[0]
            else:
                data_len = data.shape[0]

            # what do we do with multi-input data?!
            # TODO fix strange name change : vs _
            input_names = [l.name.split(":")[0] for l in model.inputs]
            # !!!!! FOLLOWING Line not tested
            input_names = [l.split("_")[0] for l in input_names]
            # split the data into batches
            for i in range(0, data_len, batch_size):
                # TODO experiment speed if with device(cpu:0)
                with tf.GradientTape() as t:
                    # for some reason requires tensor, otherwise blows up
                    # c.f. that for activations we use model.predict
                    if type(data) is dict:
                        tensor = []
                        for k in input_names:
                            tensor.append(data[k][i:min(i + batch_size, data_len)].astype('float32'))
                    else:
                        tensor = tf.convert_to_tensor(data[i:min(i + batch_size, data_len)], dtype=tf.float32)
                    current_loss = loss_(model_k(tensor))
                    dW, db = t.gradient(current_loss, [*inter_l.weights])
                score_val += fx_modulate(dW)

            # restor output function!
            if l == model.layers[-1]:
                l.activation = activation_temp

            vprint("layer:{}--{}".format(l.name, score_val.shape), verbose=verbose)
            # 2 aggragate across locations
            # 2.1 4D
            if len(score_val.shape) > 3:
                # (3, 3, 3, 64) -> for activations, it is not aggregated across data-points
                score_val = np.mean(score_val, axis=(0, 1))
                vprint("\t 4D shape:{}".format(score_val.shape), verbose=verbose)
            elif len(score_val.shape) > 2:
                score_val = np.mean(score_val, axis=(0))
                vprint("\t 3D shape:{}".format(score_val.shape), verbose=verbose)
            # 3. aggregate across datapoints

            # already produced by line 230
            # to include data point analysis, we can use persistant gradient_tape
            # compute point by point or use the loss
            # explore this issue for efficiency gain https://github.com/tensorflow/tensorflow/issues/4897

            # 4. tokenize values
            mean = np.mean(score_val, axis=1)
            omega_val[l] = mean

            vprint("\t omega_val.shape:{}".format(mean.shape), verbose=verbose)
        return omega_val


class Weight_Activations_Computer(Relevance_Computer):
    def __init__(self, model, fx_modulate=np.abs,
                 layer_start=None,
                 agg_data_points=True,
                 local=False,
                 verbose=False):
        super().__init__(model, fx_modulate, layer_start, agg_data_points, local, verbose)

    def __call__(self, data):

        model, fx_modulate, layer_start, agg_data_points, local, verbose = self.model, self.fx_modulate, self.layer_start, self.agg_data_points, self.local, self.verbose

        omega_val = {}
        for l in model.layers:

            # skips layers w/o weights
            # e.g. input/pooling
            if l.weights == []:
                omega_val[l] = np.array([])

                continue

            # 1. compute values

            # 1.1 compute activations
            model_k = tf.keras.Model(inputs=model.inputs, outputs=[l.input])
            score_val_a = model_k.predict(data)

            score_val_a = fx_modulate(score_val_a)

            # 1.2 compute

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

            # 3. aggregate across datapoints
            score_agg_a = np.mean(score_val_a, axis=0)

            # ===redundant for weights
            # 4. tokenize values
            # ===redundant for activations
            score_agg_w = np.mean(score_val_w, axis=-1)
            omega_val[l] = score_agg_a * score_agg_w
        return omega_val


def __main__():
    tf.enable_eager_execution()
    print("Core main!")
