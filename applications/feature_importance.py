from matplotlib import pyplot as plt

from aggregator_utils import *
from dginn.core import *


def dginn_global_importance_(model, X, ys, Relevance_Computer=Activations_Computer):
    X_train = X
    y_train = ys
    computer = Relevance_Computer
    compute_fx = computer(model=model, agg_data_points=True)
    dg_collections_list = []
    all_classes = np.unique(y_train).tolist()
    for cls in all_classes:
        idx_cls = np.where(y_train == cls)[0]
        # print(idx_cls[0:10])
        #     dgs_cls = extract_dgs_by_ids(relevances, idx_cls)
        dgs_cls = compute_fx(X_train[idx_cls, :])
        dg_collections_list.append(dgs_cls)
    return dg_collections_list


def dginn_global_importance(model,X,ys,Relevance_Computer=Activations_Computer,cls=None):
    dg_collections_list = dginn_global_importance_(model,X,ys,Relevance_Computer=Relevance_Computer)
    # TODO
    # (1) return the importance averaged over all class s.t.:
    # for each point the feature importance corresponds to the correct / predicted class importance
    # (2) investigate why the feature importance is the same regardless of the class
    f_nb_0 = dg_collections_list[0][model.layers[0]]
    f_nb_1 = dg_collections_list[1][model.layers[0]]
    print("dg_collections_list::Same class importance:",np.array_equal(f_nb_0,f_nb_1))
    # accumulator
#     f_nb_acc = np.zeros_like(dg_collections_list[0][model.layers[0]])
    # average over classes / should be TODO (1) from above
    if cls is None:
        f_nb_list = []
        for i in range(len(dg_collections_list)):
            f_nb_list.append(dg_collections_list[i][model.layers[0]])
        return np.mean(f_nb_list,axis=0)
    else:
        return dg_collections_list[cls][model.layers[0]]


def dginn_local_importance_(model, X, ys, Relevance_Computer=Activations_Computer):
    '''
    Return collection of dependency graphs, by class

    :param model:
    :param X:
    :param ys:
    :param Relevance_Computer:
    :return:
    '''
    X_train = X
    y_train = ys
    computer = Relevance_Computer
    compute_fx = computer(model=model, agg_data_points=False)
    dg_collections_list = []
    all_classes = np.unique(y_train).tolist()
    for cls in all_classes:
        idx_cls = np.where(y_train == cls)[0]
        # print(idx_cls[0:10])
        #     dgs_cls = extract_dgs_by_ids(relevances, idx_cls)
        dgs_cls = compute_fx(X_train[idx_cls, :])
        dg_collections_list.append(dgs_cls)
    return dg_collections_list


def dginn_local_importance(model,X,ys,Relevance_Computer=Activations_Computer,selected_classes = None):

    # Get DG collection for every class, for every point
    dg_collections_list = dginn_local_importance_(model, X, ys, Relevance_Computer=Relevance_Computer)

    # Extract feature importance for both classes.
    f_nb_0 = dg_collections_list[0][model.layers[0]]
    f_nb_1 = dg_collections_list[1][model.layers[0]]

    if selected_classes is None:
        selected_classes = [i for i in range(len(dg_collections_list))]

    f_nb_list = []

    for cls in selected_classes:
        f_nb_list.append(dg_collections_list[cls][model.layers[0]])

    return f_nb_list




# VISUALISATION FXS

def vis_feature_nb(f_nb,ax=None,figsize=None):
    if ax is None:
        import matplotlib as mpl
        if figsize is None:
            figsize = mpl.rcParams['figure.figsize']
        fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.bar(range(len(f_nb)),f_nb)
    return ax

def vis_global_unit_importance(model,dg_collections_list):
    #global neuron importance visualisation across layers
    n_layers = len(model.layers)
    n_cls = len(dg_collections_list)
    fig,axes = plt.subplots(n_layers,n_cls,figsize=(10,5))
    for j,l in enumerate(model.layers):
        for i in range(2):
            ax = axes[j][i]
            f_nb = dg_collections_list[i][l]
            vis_feature_nb(f_nb,ax=ax)


## NO USED
def dginn_importance(model, X, ys, Relevance_Computer=Activations_Computer):
    # 2 options
    # 1. compute per data-point, do fancy aggregation,
    # 2. average based on something
    # input_layers = get_input_layers(model, include_adjacent=True)
    computer = Relevance_Computer(model, fx_modulate=np.abs, agg_data_points=True)

    agg = get_count_aggregators_per_class(X, ys, model, n_samples=100, Relevance_Computer=computer)


def main():
    # SCRAP
    # X_train = X
    y_train = Y_train
    computer = Activations_Computer
    computer = Weight_Activations_Computer
    dgs = compute_dg_per_datapoint(X_train, model, computer)
    dg_collections_list = []
    all_classes = np.unique(y_train).tolist()
    for cls in all_classes:
        idx_cls = np.where(y_train == cls)[0]
        # print(idx_cls[0:10])
        dgs_cls = extract_dgs_by_ids(dgs, idx_cls)
        dg_collections_list.append(dgs_cls)


if __name__ == '__main__':
    main()

