import tensorflow as tf

if __name__=='__main__':
    tf.enable_eager_execution()

from demo.data_loaders.uci_datasets import *
from dginn.core import *
from aggregator_utils import compute_dg_per_datapoint, extract_dgs_by_ids
from demo.amy_demo.model_build import *
from applications.feature_importance import dginn_local_importance, dginn_global_importance

from adversarial_explanations.evaluate import  plot_ranking_histograms

# Load data
sensitive_feature_name = None # TODO: experiment with sensitive features
file_path = "/Users/AdminDK/Desktop/Datasets/adult/" # TODO: load from config file
Xtr, Xts, ytr, yts, _, _ = get_adult(None, file_path=file_path)
X_train, _, y_train, _ = prep_data(Xtr, Xts, ytr, yts, verbose=1)

# Build basic model with a set of Dense layers
model = build_model()

# Fit model to the data
fit_model(model, X_train, y_train, verbose=1)

# TODO: Generate adversarially-trained model

# Extract feature importances using different methods
feature_importance_methods = [dginn_local_importance]

# Specify feature to be analysed
selected_feature = 10

# Run histogram plotting
plot_ranking_histograms([model], Xts, selected_feature, yts, attribution_methods=dginn_local_importance)