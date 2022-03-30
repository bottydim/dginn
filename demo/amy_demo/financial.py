import tensorflow as tf

if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()

from applications.feature_importance import dginn_local_importance
from demo.amy_demo.adv_model_generator import adversarial_explanation_wrapper
from evaluate import *
from demo.amy_demo.model_build import *
from demo.data_loaders.uci_datasets import *


def main():
    # Load data

    dataset_id = 1
    z_idx = adult_sensitive_features_dict["age"]
    data_list = get_data_list()
    data = data_list[dataset_id]
    X_test, X_train, Y_test, Y_train = data

    # Build basic model with a set of Dense layers
    model = build_model()

    # Fit model to the data
    from pathlib import Path
    path_dir = Path(os.path.dirname(os.path.abspath("__file__"))).parents[0].parents[0]
    path_dir = path_dir / "temp_models"
    print(path_dir)
    model_save_path = path_dir / "uci_model_dataset_{}.h5".format(dataset_id)
    modified_save_path = path_dir / "uci_modified_dataset_{}.h5".format(dataset_id)

    # convert to strings to avoid keras filepath.endswith(."h5") line
    model_save_path = str(model_save_path)
    modified_save_path = str(modified_save_path)
    if not os.path.exists(model_save_path):
        # Train model
        fit_model(model, X_train, Y_train, verbose=1)
        # Generate adversarially-trained model
        model_modified = adversarial_explanation_wrapper(X_train, Y_train, model, z_idx=z_idx, e_alpha=0.25)
        print("Generated model modified")
        if not os.path.exists(path_dir):
            try:
                path = path_dir
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)
        model.save_weights(model_save_path)
        print(modified_save_path)
        model_modified.save_weights(modified_save_path)

    else:
        model.load_weights(model_save_path)
        model_modified = tf.keras.models.clone_model(model)
        model_modified.load_weights(modified_save_path)

    # from functools import partial
    # dginn_grad = partial(dginn_local_importance,Relevance_Computer=Gradients_Computer)
    dginn_grad = dginn_local_importance

    loss = lambda x: x  # Use the identity for neuron preprocessing
    agg_data_points = False
    agg_neurons = False
    from dginn.core import Gradients_Computer, DepGraph
    grad_computer = Gradients_Computer(model, loss=loss,
                                       agg_data_points=agg_data_points,
                                       agg_neurons=agg_neurons)

    dep_graph = DepGraph(grad_computer)
    dginn_full = dep_graph.feature_importance
    # Extract feature importances using different methods
    feature_importance_methods = [saliency, gradient_x_input, integrated_grads, guided_backprop, dginn_grad, dginn_full]

    # TODO check once bug is fixed!
    feature_importance_methods = [saliency, dginn_full]
    # Compute and compare feature importances for both the original and adversarially-trained model
    models = [model, model_modified]
    X = X_train
    Y = Y_train
    inputs = tf.convert_to_tensor(X, dtype=tf.float32)
    outputs = tf.convert_to_tensor(Y, dtype=tf.float32)
    att_method_str = ["Gradients", "Gradient*Input", "Integrated Gradients", "Guided-Backprop", "DGINN local",
                      "DGINN FULL"]
    att_method_str = ["Gradients", "DGINN FULL"]
    attribution_list = generate_attribution_list(models, inputs, z_idx, ys=outputs,
                                                 attribution_methods=feature_importance_methods, )

    fig = plot_ranking_histograms_att(attribution_list, att_method_str, num_f=X_train.shape[1], z_idx=z_idx,
                                      models_str=["Original", "Modified"])

    fig.show()


if __name__ == '__main__':
    main()
