import tensorflow as tf

if __name__=='__main__':
    tf.enable_eager_execution()

from demo.data_loaders.uci_datasets import *
from dginn.core import *
from aggregator_utils import compute_dg_per_datapoint, extract_dgs_by_ids
from demo.amy_demo.model_build import *


# Load data
sensitive_feature_name = None # TODO: experiment with sensitive features
file_path = "/Users/AdminDK/Desktop/Datasets/adult/" # TODO: load from config file
Xtr, Xts, ytr, yts, _, _ = get_adult(None, file_path=file_path)
X_train, _, y_train, _ = prep_data(Xtr, Xts, ytr, yts, verbose=1)

# Build basic model with a set of Dense layers
model = build_model()

# Fit model to the data
fit_model(model, X_train, y_train, verbose=1)

# Generate adversarially-trained model


# Extract feature importances using different methods
feature_importance_methods = []

for feature_importance_method in feature_importance_methods:

    # Compute and compare feature importances for both the original and adversarially-trained model
    feature_importance_orig = ...
    feature_importance_adv = ...

    # TODO: how to compare? draw figures?