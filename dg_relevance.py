import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# this should be only in the call module, all other modules should not have it!!!
# best keep it in the main fx! 
# tf.enable_eager_execution()
config = tf.ConfigProto()
# config.gpu_options.visible_device_list = str('1')
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

global verbose
verbose = False


##########COMPUTE##########
# computes the importance scores of neurons to upper layer neurons 
# different methods 
def compute_weight(model, fx_modulate=lambda x: x, verbose=False, local=False):
    '''
    model: model analyzed
    fx_module: extra weight processing
    '''
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
        # 4. Global Aggregation: tokenize values across upper layer neurons
        if not local:
            score_agg = np.mean(score_val, axis=-1)
        else:
            score_agg = score_val
        vprint("\tomega_val.shape:{}".format(score_agg.shape), verbose=verbose)
        omega_val[l] = score_agg

    return omega_val


compute_weight_abs = lambda x: compute_weight(x, fx_modulate=np.abs)


### ACTIVATIONS

### ACTIVATIONS ALL! 
# LOCAL computaion - rename!




### ACTIVATIONS Global
def compute_activations_gen(data, fx_modulate=lambda x: x, layer_start=None,
                            agg_data_points=True,
                            verbose=False):
    """
    agg_data_points: whehter to aggragate across data points
    return: return all neuron importance for all layers starting from layer_start
    
    compute_activations = compute_activations_gen(X_test)
    DG_a_mean = relevance_select_mean(compute_activations(model),input_layer=model.layers[0])
    """

    def compute_activations_(model):
        omega_val = {}
        for l in model.layers[layer_start:]:
            vprint("layer:{}".format(l.name), verbose=verbose)
            # 1. compute values
            # l.output vs l.input
            # l.output => r[l] = NB on the same layer (which makes sense for activations)
            # conv1d_8--(2, 1170, 20)
            # 3D shape:(2, 20)
            # omega_val.shape:(20,)
            # CHALLENGE => it doesn't make sense for the last layer
            # layer:targetPower--(1, 23)
            # omega_val.shape:(23,) => where instead we want the importance
            # l.input  =>  r[l] = NB of the
            # conv1d_8--(2, 1199, 1) \\ shape of previous layer!
            # 3D shape:(2, 1)
            # omega_val.shape:(1,)
            # l.input is NOT working for multi input
            # crash @concatenate

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

    return compute_activations_


### GRADS

def compute_grads_helper(model, data, loss_, fx_modulate):
    # TODO
    raise NotImplemented
    return omega_val


def compute_grads_gen(data, loss_=lambda x: tf.reduce_sum(x[:, :]), fx_modulate=lambda x: x, verbose=False,
                      batch_size=128):
    """
    General idea: go through all layers, compute the derivatives of the upper layer neurons wrt current 
    
    e.g. 
    compute_grads = compute_grads_gen(X_test,loss_fx)
    grads = compute_grads(model)
    """

    def compute_grads(model):
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

    return compute_grads


### Weights * activations


def compute_weight_activations_gen(data, fx_modulate=lambda x: x, verbose=verbose):
    def compute_weight_activations_(model):
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

    return compute_weight_activations_


##### Increase Granularity: Weights Local

def compute_weight_per_channel(model, fx_modulate=lambda x: x):
    '''
    equivalent to compute_weight(model,fx_modulate=lambda x:x,verbose=False,local=True)
    '''
    omega_val = {}
    for l in model.layers:
        # skips layers w/o weights 
        # e.g. input/pooling 
        if l.weights == []:
            #             print(l.name)
            omega_val[l] = np.array([])
            continue
        # print("layer:{}".format(l.name))
        # 1. compute values
        #         print("w.shape:{}".format(l.weights[0].shape))
        score_val = l.weights[0][:, :]
        score_val = fx_modulate(score_val)

        # 1.1 aggragate across locations
        if len(score_val.shape) > 3:
            #             print("\tshape:{}")
            #             print("\tscore_val.shape:{}".format(score_val.shape))
            score_val = np.mean(score_val, axis=(0, 1))
        # 2. aggregate across datapoints
        # ===redundant for weights
        # 3. tokenize values across upper layer neurons
        #         score_agg = np.mean(score_val,axis=-1)
        score_agg = score_val
        #         print("\tomega_val.shape:{}".format(score_agg.shape))
        omega_val[l] = score_agg

    return omega_val


######## Combine Compute FX
def combine_compute_fx(relevances_a, relevances_b):
    '''
    
    e.g.
    compute_fx = compute_activations_gen(data_set,fx_modulate=np.abs,verbose=False)
    relevances_a = compute_fx(model)
    relevances_w = compute_weight(model,fx_modulate=np.abs)

    compute_fx = compute_weight_activations_gen(data_set,fx_modulate=np.abs,verbose=False)
    relevances_wa = compute_fx(model)

    compute_fx = compute_grads_gen(data_set,fx_modulate=np.abs,verbose=False)
    relevances_g = compute_fx(model)

    relevances_wa_app = combine_compute_fx(relevances_a,relevances_w)
    relevances_ga_app = combine_compute_fx(relevances_a,relevances_g)
    '''
    relevances_ab_app = {}
    for k, v in relevances_a.items():
        #         print(k.name,type(relevances_a[k]),type(relevances_b[k]))
        if k.weights == []:
            relevances_ab_app[k] = np.array([])
            continue
        relevances_ab_app[k] = relevances_a[k] * relevances_b[k]
    return relevances_ab_app


######### DEPENDENCY

# TODO abstract out as fx!
# one way to do is to make all computes classes 
# give them interfaces

# COULD be equivalent to compute_weight_per_channel  ???!! 
def compute_dependency_weight(model, fx_modulate=lambda x: x):
    '''
    - does not aggragate across upper layer! 
    equivalent to compute_weight(model,fx_modulate=lambda x:x,verbose=False,local=True)
    
    
    '''
    omega_val = {}
    for l in model.layers:
        # skips layers w/o weights 
        # e.g. input/pooling 
        if l.weights == []:
            #             print(l.name)
            omega_val[l] = np.array([])
            continue
        # print("layer:{}".format(l.name))
        # 1. compute values
        #         print("w.shape:{}".format(l.weights[0].shape))
        score_val = l.weights[0][:, :]
        score_val = fx_modulate(score_val)

        #         print("\tscore_val.shape:{}".format(score_val.shape))
        # 2 aggragate across locations
        shape_len = len(score_val.shape)
        # 2.1 4D input (c.f. images)
        if shape_len > 3:
            score_val = np.mean(score_val, axis=(0, 1))
        # 2.2 aggregate across 1D-input
        elif shape_len > 2:
            score_val = np.mean(score_val, axis=(0))
            # 3. aggregate across datapoints
            # ===redundant for weights
            # 4. tokenize values across upper layer neurons
        #         score_agg = np.mean(score_val,axis=-1)
        score_agg = score_val
        #         print("\tomega_val.shape:{}".format(score_agg.shape))
        omega_val[l] = score_agg

    return omega_val


##########GENERATE DG##########
# tokenize


# functions, which handle the neuron relevance selection
def percentage_threshold(relevance, t):
    r = relevance
    # sort args based on val, reverse, take %
    return list(reversed(np.argsort(r)))[:int(len(r) * t)]


def select_random(relevance, threshold):
    size = len(relevance)
    return np.random.choice(size, threshold)


def relevance_select_(omega_val, input_layer, select_fx_):
    relevant = {}
    for l, relevance in omega_val.items():
        if type(input_layer) is list and l in input_layer:
            relevant[l] = range(len(relevance))
        elif l == input_layer:
            #             print(len(relevance),relevance)
            relevant[l] = range(len(relevance))
        # process layers without weights
        elif l.weights == []:
            relevant[l] = []
        else:

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


##########EVAL HELPERS##########
def get_time_hhmmss(start):
    import time
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str


# inverse mask of the Dependency Graph:
# turn-off all neurons, which are not part of the DG, and execute the network!
# the same as running just the DG (dependency graph)
def get_DG_exec(model, dep_graph):
    temp_store = {}
    for l in model.layers:
        import time
        start = time.time()
        # skip pooling & flatten & concatenate
        if len(l.weights) == 0:
            continue
        # drop irrelevant
        w, b = l.weights
        # this way we create a copy? [:,;]?
        temp_store[l] = w[:, :], b[:]
        # gen all neuron indices in the current layer
        # and subtract from them all relevant indices (neurons in DG)
        # -2 in order to work for both convolutional[-2] & dense layers[0]
        irrelevant = set(range(w.shape[-2])) - set(dep_graph[l])
        neurons = list(irrelevant)

        target = w.numpy()
        n = neurons
        # skip layers which do not require any weight changes
        # observe the BUG! 
        if len(n) == 0:
            continue
        # 4D convolutional
        if len(w.shape) > 3:
            target[:, :, n, :] = 0
        elif len(w.shape) > 2:
            target[:, n, :] = 0
            target_shape = target.shape
        else:
            target[n, :] = 0
            target_shape = target.shape

        tf.assign(w, target)
        # don't touched the bias, as it is like another neuron for the upper layer neuron
    #                 print(b.shape,n)
    #                 tf.assign(b[n], np.zeros(b.shape))
    #         print(l.name,get_time_hhmmss(start))
    return model, temp_store


def recover_model(model, temp_store):
    for l in model.layers:
        # skip pooling & flatten
        if len(l.weights) == 0:
            continue
        w, b = l.weights
        temp_w, temp_b = temp_store[l]
        tf.assign(w, temp_w)
    return model


##########EVAL##########


#### EXTREMELY INEFFICIENT WAY TO EVAL!
def evaluate_dg_gen(model, data):
    X_, y_ = data

    def eval_dep_graph(dep_graph):
        temp_store = {}
        for l in model.layers:
            # skip pooling & flatten
            if len(l.weights) == 0:
                continue
            # drop irrelevant
            w, b = l.weights
            temp_store[l] = w[:, :], b[:]
            # gen all neuron indices and subtract from them all relevant indices
            # -2 in order to work for both convolutional[-2] & dense layers[0]
            irrelevant = set(range(w.shape[-2])) - set(dep_graph[l])
            neurons = list(irrelevant)

            # INEFFICIENT!! 
            for n in neurons:
                if len(w.shape) > 2:
                    target = w[:, :, n, :]
                    target_shape = target.shape
                    tf.assign(target, np.zeros(target.shape[:], ))
                else:
                    tf.assign(w[n, :], np.zeros(w.shape[1]))
                    # don't touched the bias, as it is like another neuron for the upper layer neuron
                    #                 print(b.shape,n)
                    #                 tf.assign(b[n], np.zeros(b.shape))
        score = model.evaluate(X_, y_, verbose=0, batch_size=256)

        for l in model.layers:
            # skip pooling & flatten
            if len(l.weights) == 0:
                continue
            w, b = l.weights
            temp_w, temp_b = temp_store[l]
            tf.assign(w, temp_w)
        return score

    return eval_dep_graph


evaluate_dep_graph = lambda model, DG: evaluate_dg_gen(model, (X_test, Y_test))(DG)
evaluate_dep_graph_train = lambda model, DG: evaluate_dg_gen(model, (X_train, Y_train))(DG)


# runs evalaution of the computed dependecy graphs computed with various compute functions across different data-sets.
def evaluate_across_classes_with_threshold(model, data_sets, store_DGs=False, t_lim=None, ):
    def loss_fx(x):
        return tf.reduce_mean(x[:, :])

    DGs = []

    #     import pdb; pdb.set_trace()
    train_data_sets, eval_data_sets = data_sets
    print(len(train_data_sets), train_data_sets[0][0].shape)
    print(len(eval_data_sets))
    timer = Timer()
    thresholds = [(i + 1) / 10.0 for i in range(10)][::-1]
    j_len = len(train_data_sets)
    i_len = 3
    k_len = len(eval_data_sets)
    t_len = len(thresholds)
    results = np.zeros((k_len, j_len, i_len, t_len, 2))
    for j, data_set in enumerate(train_data_sets):

        X_temp, y_temp = data_set
        #         compute_activations = compute_activations_gen(X_temp)
        compute_activations_abs = compute_activations_gen(X_temp, fx_modulate=np.abs)
        #         compute_grads = compute_grads_gen(X_temp,loss_fx)
        compute_grads_abs = compute_grads_gen(X_temp, loss_fx, fx_modulate=np.abs)
        #         compute_wa = compute_weight_activations_gen(X_temp)
        compute_wa_abs = compute_weight_activations_gen(X_temp, np.abs)

        #         compute_fxs = [compute_weight, compute_weight_abs,compute_activations,compute_activations_abs,
        #                       compute_grads,compute_grads_abs,compute_wa,compute_wa_abs]
        compute_fxs = [compute_activations_abs,
                       compute_grads_abs, compute_wa_abs]
        #         compute_fxs = [compute_wa_abs]
        print("Generate Compute FXs-{}".format(timer.get_time_hhmmss()))
        # eval

        DGs.append([])
        for i, compute_fx in enumerate(compute_fxs):
            relevances = compute_fx(model)
            DGs[j].append([])
            print("COMPUTATION:", timer.get_time_hhmmss())
            for t_, t in enumerate(thresholds):
                DG = relevance_select(relevances, input_layer=model.layers[0], threshold=t)
                if store_DGs:
                    DGs[j][i].append(DG)
                dg_exec, tmp = get_DG_exec(model, DG)
                for k, eval_ds in enumerate(eval_data_sets):
                    eval_dg = evaluate_dg_gen(model, eval_ds)
                    score = dg_exec.evaluate(*eval_ds, batch_size=256, verbose=0)

                    results[k, j, i, t_] = score
                    assert np.all(results[k, j, i, t_] == score)
                # print(timer.get_time_hhmmss())
                # print info
                recover_model(model, tmp)
                fx = compute_fxs[i]
                name = "_".join(str(fx).split(" ")[1].split("_")[1:2])
                print("{}-#{}-{}-ts={}".format(name, j, i, t))
                print(timer.get_time_hhmmss())
    if store_DGs:
        return results, DGs
    else:
        return results


# LEGACY
def analyzer_gen(model):
    def analyze_dg(compute_fx, relevance_select_fx):

        dg_vals = compute_fx(model)
        value = list(dg_vals.values())
        for i in range(len(value)):
            if len(value[i]) == 0:
                continue
            mean = value[i]
            order = np.flip(np.argsort(np.abs(mean)))
            #     plt.figure()
            plt.scatter(np.arange(mean.shape[0]), np.abs(mean[order]))
        DG = relevance_select_fx(compute_fx(model), input_layer=model.layers[0])
        report_eval_dep_graph(model, DG)
        return DG

    return analyze_dg


######## Data Set Processing

### CIFAR-specific
def get_data_sets_cls(X, Y, d_lim=None):
    data_sets = []
    for i in range(Y.shape[1]):
        idx = np.where(Y[:, i] == 1)[0]
        X_cls_i = X[idx]
        y_cls_i = Y[idx]
        #                 print(X_cls_i.shape,y_cls_i.shape)
        data_sets.append((X_cls_i[0:d_lim], y_cls_i[0:d_lim]))
    return data_sets


def get_data_sets(data_sets, cls_eval=True, d_lim=None):
    '''
    data_sets = [(X_train[0:256],Y_train[0:256]),(X_test[0:256],Y_test[0:256])]
    a,b = get_data_sets(data_sets)
    assert len(a) == 11 and len(b) == 22
    for ds in a:
        assert len(ds)==2
    for ds in b:
        assert len(ds)==2
    '''
    data_sets = data_sets[0:d_lim]
    train_data_sets = data_sets[0:1]
    train_data_cls = cls_eval
    # gen data_sets
    # FIX, these are global vars X_train,Y_train
    if train_data_cls:
        train_data_sets += get_data_sets_cls(X_train, Y_train)
    eval_data_sets = [_ for _ in train_data_sets]
    if len(data_sets) > 1:
        eval_data_sets += [data_sets[1]]
    # data_sets = [(X_train[:100],Y_train[:100]),(X_test[:100],Y_test[:100])]

    test_data_cls = cls_eval
    if test_data_cls:
        X_test = data_sets[1][0]
        Y_test = data_sets[1][1]
        for i in range(Y_test.shape[1]):
            idx = np.where(Y_test[:, i] == 1)[0]
            X_cls_i = X_test[idx]
            y_cls_i = Y_test[idx]
            #             print(X_cls_i.shape,y_cls_i.shape)
            eval_data_sets.append((X_cls_i, y_cls_i))
    return train_data_sets, eval_data_sets


def merge_data_sets(cls_ds_list, id_list, d_lim=None):
    '''
    data_sets_list: list of (X,Y) typles
    '''
    output_ds = []

    for cls_ds in cls_ds_list:
        data_sets_merged = []
        for i in id_list:
            data_sets_merged += cls_ds[i]
        # irrelevant of len(id_list!)
        try:
            # get every second element b.c. there are X & Y
            X_ = np.vstack(data_sets_merged[::2])
            Y_ = np.vstack(data_sets_merged[1::2])
            output_ds.append((X_, Y_))
        except:
            print(X.shape)
            for x in data_sets_merged:
                print(x.shape)
            raise "ERROR"
    return output_ds


##########UTILS##########  

# relabel the key of the Dependency Graphs 
def relabel(DGs):
    DG_relabelled = {}
    for j in range(shapes[1]):
        for i in range(shapes[2]):
            for t_ in range(shapes[3]):
                DG = DGs[j][i][t_]
                for k, l in enumerate(model.layers):
                    try:
                        temp = DG[l]
                        DG_relabelled[k] = temp
                    except:
                        print(j, i, t_)
                        return
    return DG_relabelled


def vprint(*args, verbose=False):
    '''
    verbose = True
    args = 2,"abc","{}".format(2)2 abc 2
    vprint(*args,verbose=True)
    print(*args)
    print(2,"abc","{}".format(2))
    vprint(*args,verbose=0)
    '''
    # TODO fix bug when verbose is arg,not kwarg
    # eg. vprint(*args,verbose)
    if verbose:
        print(*args[:])


##########COMPUTE UTILS##########
# c.f. = confer =  compare

### counters
def cf_dgs(dg_a, dg_b):
    '''
    count number of shared neurons
    '''
    total = 0
    for (k, v) in dg_a.items():
        a = set(v)
        b = set(dg_b[k])
        r = len(a.intersection(b))
        total += r
    return total


# count unique!
def cf_dgs_unique(dg_a, dg_b):
    total = 0
    for (k, v) in dg_a.items():
        a = set(v)
        b = set(dg_b[k])
        c = a - b
        r = len(c)
        total += r
    return total


### DG extractors
def cf_dgs_intersect(dgs):
    '''
    dgs: list of dependency graphs
    return: intersection b/t all dgs
    '''
    total = 0
    import copy
    dg_ = copy.copy(dgs[0])
    for dg_b in dgs[1:]:
        for (k, v) in dg_.items():
            a = set(v)
            b = set(dg_b[k])
            c = a.intersection(b)
            dg_[k] = c
    return dg_


def cf_dgs_union(dgs):
    '''
    dgs: list of dependency graphs
    return: union b/t all dgs
    '''
    total = 0
    import copy
    dg_ = copy.copy(dgs[0])
    for dg_b in dgs[1:]:
        for (k, v) in dg_.items():
            a = set(v)
            b = set(dg_b[k])
            c = a.union(b)
            dg_[k] = c
    return dg_


def cf_dgs_diff(dg_a, dg_b, skip_layers=[]):
    '''
    dg_a: minuend
    dg_b: subtrahend
    skip_layers: input layers to skip, if execution of the graph is required! 
    '''
    import copy
    dg_ = copy.copy(dg_a)
    for (k, v) in dg_a.items():
        if k in skip_layers:
            continue
        a = set(v)
        b = set(dg_b[k])
        c = a - b
        dg_[k] = c
    return dg_


def report_eval_dep_graph(model, dep_graph):
    print(evaluate_dep_graph(model, dep_graph))
    print(count_dep_graph(model, dep_graph))


def len_dep_graph(DG):
    total = 0
    for l, neurons in DG.items():
        total += len(neurons)
    return total


def count_dep_graph(model, dep_graph):
    total_slct, total_all = 0, 0
    for l in model.layers:
        if len(l.weights) == 0:
            continue
        n_slct = len(dep_graph[l])
        n_total = len(omega_val[l])
        print("{}/{}={:2.1f}%".format(n_slct, n_total, 100 * n_slct / float(n_total)))
        total_slct += n_slct
        total_all += n_total
    return total_slct, total_all


def visualise_layers_DG(DG, DG_full, model, percentage=False, skip_layers=[]):
    layer_vals = []
    layer_size = []
    layer_names = []
    for l in model.layers:
        if l.name in skip_layers:
            continue
        layer_vals.append(len(DG[l]))  #
        layer_names.append(l.name)
        layer_size.append(len(DG_full[l]))
    if percentage:
        layer_vals = np.array(layer_vals) / np.array(layer_size) * 100
    N = len(layer_vals)
    fig, ax = plt.subplots(figsize=(10, 5))
    ind = np.arange(N)  # the x locations for the groups
    width = 0.50  # the width of the bars
    p1 = ax.barh(ind, layer_vals, width, )
    if percentage:
        ax.set_xlim(0, 100)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names)


# ax.invert_yaxis()

def get_input_layers(model, include_adjacent=True):
    if include_adjacent:
        input_layers = []
        for l in model.layers:
            if l.input in model.inputs:
                input_layers.append(l)
    else:
        input_layers = list(filter(lambda x: type(x) == type(model.layers[0]), model.layers))
    return input_layers


def view_input_layers(relevances, input_layers):
    '''
    return: the importance of the input layers
    e.g. 
    aggPower [33.246933]
    aggDiffWave [0.05136125]
    aggNoMinWave [0.23131992]
    '''
    for l in input_layers:
        print(l.name, relevances[l])


def visualise_kernel_dependence(importance_kernels_1, sharey=False, log=False):
    n_rows = 10
    n_cols = int(int(importance_kernels_1.shape[-1]) / n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), sharey=sharey)
    for i in range(n_rows):
        for j in range(n_cols):
            idx = n_cols * i + j
            y_axis_ = np.abs(importance_kernels_1[:, idx])
            line = axes[i, j].bar(np.arange(y_axis_.shape[0]), y_axis_, log=log)
            axes[i, j].set_title("Kernel #:{}".format(idx))


def layer_importance_strategies(relevance_strategies, relevance_names, logy=False, sharey=False):
    """
    relevance_strategies: a list of 
    """
    # TODO fix number! to be dynamic based on input
    n_rows = 3
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), sharey=sharey)
    for i, r in enumerate(relevance_strategies):
        x = i // n_cols
        y = i % n_cols
        ax = axes[x, y]
        view_layer_importance(r, model, logy=logy, ax=ax)
        ax.set_title(relevance_names[i])
    plt.tight_layout()


def run_dgs(DG_eval, data_sets, model):
    """
    return: a list for each DG where each element is the output of the corresponding DG on a particular data_set
    """
    results_all = []
    my_timer = Timer()
    for i, DG in enumerate(DG_eval):
        results_list = []
        print("DG_{}_{}".format(i, my_timer.get_time_hhmmss()))
        for j in range(len(data_sets)):
            X_eval = data_sets[j]
            dg_exec, tmp = get_DG_exec(model, DG)
            # FIX: y_eval is a global var! 
            # ?? batch_size=WHY? y_eval.shape[0]
            results = dg_exec.predict(X_eval, batch_size=128, verbose=0)
            recover_model(model, tmp)

            results_list.append(results)
        results_all.append(results_list)
    return results_all


###### Predictions Analysis  (potentially INFORMETIS specific)

def get_overall_predictions(results_list, results_orig):
    pred_orig = argmax_predictions(results_orig)
    pred_dg = argmax_predictions(results_list)
    return pred_dg, pred_orig


def mean_predictions(results_list):
    """
    results_list: list of multiple DG run results
    return: mean prediction for each DG
    """
    pred = []
    N = len(results_list)
    for i in range(N):
        results_mean = np.mean(results_list[i], axis=0)
        pred.append(np.argmax(results_mean))
    return pred


def argmax_predictions(results_list):
    """
    results_list: list of multiple DG run results
    return: argmax prediction per sample for each DG
    """
    pred = []
    N = len(results_list)
    for i in range(N):
        results = np.argmax(results_list[i], axis=1)
        bins = np.bincount(results, minlength=results_list[i].shape[1])
        pred.append(np.argmax(bins))
    return pred


def get_pred_dist(results_list):
    """
    return: distribution of predictions in hist form c.f. bins
            based on argmax_predictions
    """
    pred_dist = []
    N = len(results_list)
    for i in range(N):
        results = np.argmax(results_list[i], axis=1)
        pred_dist.append(np.unique(results, return_counts=True))
    return pred_dist


def total_prediction_count(results_list):
    """
    return: counts # predictions for each class across result_list of DGs
    """
    num_cls = results_list[0].shape[1]
    acc_array = np.zeros(num_cls)
    N = len(results_list)
    for i in range(N):
        results = np.argmax(results_list[i], axis=1)
        bins = np.bincount(results, minlength=num_cls)
        acc_array += bins
    return acc_array


def count_correct_pred(results_list, current_labels):
    correct_counter = []
    for i in range(len(results_list)):
        results = np.argmax(results_list[i], axis=1)
        bins = np.bincount(results, minlength=results_list[i].shape[1])
        dg_correct = bins[current_labels[i]]
        correct_counter.append(dg_correct)
    return correct_counter


#### Secondary Analysis

# un-tested functions

def plot_input_importance(relevance_list, input_layers, data_set_cls_dict):
    bars = {k.name: [] for k in input_layers}
    for r in relevance_list:
        for l in input_layers:
            bars[l.name].extend(r[l])
    N = len(relevance_list)
    fig, ax = plt.subplots()
    plots = []
    for i, k in enumerate(bars.keys()):
        ind = np.arange(N) * 2  # the x locations for the groups
        width = 0.50  # the width of the bars
        p1 = ax.bar(ind + i * width, bars[k], width, bottom=0)
        plots.append(p1[0])
    ax.set_title('Relevance by input layer and class type')
    ax.set_xticks(ind + width / 3)
    class_names = [cls for cls in data_set_cls_dict.keys()]
    ax.set_xticklabels(class_names)
    input_layer_labels = [l for l in bars.keys()]
    ax.legend(plots, input_layer_labels)


# TODO Remove
#### UTILS 

import time


# %load -r 78-111 /home/btd26/xai-research/xai/xai/utils/utils.py
class Timer:
    '''
    # Start timer
  my_timer = Timer()

  # ... do something

  # Get time string:
  time_hhmmss = my_timer.get_time_hhmmss()
  print("Time elapsed: %s" % time_hhmmss )

  # ... use the timer again
  my_timer.restart()

  # ... do something

  # Get time:
  time_hhmmss = my_timer.get_time_hhmmss()

  # ... etc
    '''

    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_time_hhmmss(self):
        end = time.time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str
