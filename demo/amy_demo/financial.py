import tensorflow as tf

if __name__=='__main__':
    tf.enable_eager_execution()

from demo.data_loaders.uci_datasets import *
from dginn.core import *
from aggregator_utils import compute_dg_per_datapoint, extract_dgs_by_ids
from demo.amy_demo.model_build import *
from .adv_model_generator import adversarial_explanation_wrapper

# Load data
sensitive_feature_name = None # TODO: experiment with sensitive features
file_path = "/Users/AdminDK/Desktop/Datasets/adult/" # TODO: load from config file
Xtr, Xts, ytr, yts, _, _ = get_adult(None, file_path=file_path)
X_train, _, Y_train, _ = prep_data(Xtr, Xts, ytr, yts, verbose=1)

# Build basic model with a set of Dense layers
model = build_model()

# Fit model to the data
fit_model(model, X_train, Y_train, verbose=1)

# Generate adversarially-trained model
z_idx = adult_sensitive_features_dict["age"]
model_ae = adversarial_explanation_wrapper(X_train, Y_train, model, z_idx=z_idx, e_alpha=0.25)

# Extract feature importances using different methods
feature_importance_methods = []


def main():
    e

if __name__ == '__main__':
    main