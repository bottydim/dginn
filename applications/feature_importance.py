from dginn.core import *
from aggregator_utils import *

def dginn_importance(model, X, ys, Relevance_Computer=Activations_Computer):
    # 2 options
    # 1. compute per data-point, do fancy aggregation,
    # 2. average based on something
    # input_layers = get_input_layers(model, include_adjacent=True)
    computer = Relevance_Computer(model, fx_modulate=np.abs, agg_data_points=True)

    agg = get_count_aggregators_per_class(X, ys, model, n_samples=100, Relevance_Computer=computer)
