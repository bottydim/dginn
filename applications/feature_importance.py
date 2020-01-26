from matplotlib import pyplot as plt

from aggregator_utils import *
from dginn.core import *


def dginn_importance(model, X, ys, Relevance_Computer=Activations_Computer):
    # 2 options
    # 1. compute per data-point, do fancy aggregation,
    # 2. average based on something
    # input_layers = get_input_layers(model, include_adjacent=True)
    computer = Relevance_Computer(model, fx_modulate=np.abs, agg_data_points=True)

    agg = get_count_aggregators_per_class(X, ys, model, n_samples=100, Relevance_Computer=computer)


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


# VISUALISATION FXS

def vis_global_unit_importance(model, dg_collections_list):
    # global neuron importance visualisation across layers
    n_layers = len(model.layers)
    n_cls = len(dg_collections_list)
    fig, axes = plt.subplots(n_layers, n_cls, figsize=(10, 5))
    for j, l in enumerate(model.layers):
        for i in range(2):
            ax = axes[j][i]
            f_nb = dg_collections_list[i][l]
            ax.bar(range(len(f_nb)), f_nb)


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

