import tensorflow as tf

if __name__=='__main__':
    tf.enable_eager_execution()

from demo.data_loaders.uci_datasets import *
from dginn.core import *
from aggregator_utils import compute_dg_per_datapoint, extract_dgs_by_ids
from applications.feature_importance import dginn_local_importance
from applications.feature_importance import dginn_local_importance, dginn_global_importance
from demo.amy_demo.adv_model_generator import adversarial_explanation_wrapper
from evaluate import *
from demo.amy_demo.model_build import *
from matplotlib import pyplot as plt
def main():
    # Load data

    dataset_id = 1
    data_list = get_data_list()
    data = data_list[dataset_id]
    X_test, X_train, Y_test, Y_train = data

    # Build basic model with a set of Dense layers
    model = build_model()

    # Fit model to the data
    fit_model(model, X_train, Y_train, verbose=1)

    # Generate adversarially-trained model
    z_idx = adult_sensitive_features_dict["age"]
    model_modified = adversarial_explanation_wrapper(X_train, Y_train, model, z_idx=z_idx, e_alpha=0.25)
    print("model modified")

    # Extract feature importances using different methods
    feature_importance_methods = [saliency, gradient_x_input, integrated_grads, guided_backprop, dginn_local_importance]

    # Compute and compare feature importances for both the original and adversarially-trained model
    models = [model,model_modified]
    X = X_train
    Y = Y_train
    inputs = tf.convert_to_tensor(X, dtype=tf.float32)
    outputs = tf.convert_to_tensor(Y, dtype=tf.float32)
    fig = plot_ranking_histograms(models, inputs, z_idx, ys=outputs, num_f=X_train.shape[1], title="",
                            attribution_methods=feature_importance_methods)
    fig.show()


if __name__ == '__main__':
    main()

