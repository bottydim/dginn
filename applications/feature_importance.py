from dginn.core import *
from aggregator_utils import *

def dginn_importance(model, X, ys, Relevance_Computer=Activations_Computer):
    # 2 options
    # 1. compute per data-point, do fancy aggregation,
    # 2. average based on something
    # input_layers = get_input_layers(model, include_adjacent=True)
    computer = Relevance_Computer(model, fx_modulate=np.abs, agg_data_points=True)

    agg = get_count_aggregators_per_class(X, ys, model, n_samples=100, Relevance_Computer=computer)


if __name__ == '__main__':
    #SCRAP
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