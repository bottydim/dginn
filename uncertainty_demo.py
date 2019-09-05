import numpy as np
from mnist_loader import load_mnist, build_sample_model
from dataset_utils import filter_dataset
from data_visualizers import visualize_samples
from aggregator_utils import get_count_aggregators, compute_dg
from core import *



def sort_uncertain_points(query_x, train_x, train_y, model, n_samples=100):

    # Run samples through model to get predicted labels
    predictions = np.argmax(model.predict(query_x), axis=1)

    # Create aggregators from the training samples
    aggregators = get_count_aggregators(train_x, train_y, model, n_samples)

    similarities = {}
    compute_fx = Activations_Computer(model=model, agg_data_points=True)
    for i, x_sample in enumerate(query_x):

        print("Iteration ", i)

        # Compute dep. graph of new sample
        x_sample = np.expand_dims(x_sample, axis=0)
        dg_query = compute_dg(x_sample, compute_fx)

        # Obtain the sample predicted label
        y_pred = predictions[i]

        # Compute similarity of the test point to the sampled points
        similarities[i] = aggregators[y_pred].similarity(dg_query)


    # Sort points by their similarity
    sorted_keys = sorted(similarities, key=similarities.get)
    sorted_vals = [query_x[i] for i in sorted_keys]

    # Extract least similar 30 points
    sorted_vals = sorted_vals[:40]

    # Visualise samples
    # Idea: samples with lower similarity will seem stranger
    visualize_samples(sorted_vals)






def main():
    """
    Script demonstrating uncertainty functionality offered by dep. graphs.

    Sorts MNIST points by their uncertainty and prints them out.

    Idea: high uncertainty points are "strange" and seem atypical, compared to training data
    """

    # Load dataset
    train_x, train_y, test_x, test_y = load_mnist()

    # Filter out subset of classes
    selected_classes = [0, 1, 2, 3]
    train_x, train_y = filter_dataset((train_x, train_y), selected_classes)
    test_x, test_y = filter_dataset((test_x, test_y), selected_classes)

    # Create model
    input_shape = train_x.shape[1:]
    model = build_sample_model(input_shape)

    # Train model
    model.fit(x=train_x, y=train_y, epochs=2)

    # Select points to inspect
    selected_points = test_x[:100]

    # Visualise points, sorted by their uncertainty
    sort_uncertain_points(selected_points, train_x, train_y, model, n_samples=100)


main()