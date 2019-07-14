import numpy as np
from mnist_loader import load_mnist, build_sample_model
from dataset_utils import filter_dataset
from data_visualizers import visualize_samples
from aggregator_utils import get_count_aggregators, compute_dg








def sort_uncertain_points(x_samples, model, n_samples=100):


    # Run samples through model to get predicted labels
    predictions = np.argmax(model.predict(x_samples), axis=1)

    aggregators = get_count_aggregators(x_samples, predictions, n_samples)

    similarities = {}

    for i, x_sample in enumerate(x_samples):

        print("Iteration ", i)

        # Compute dep. graph of new sample
        x_sample = np.expand_dims(x_sample, axis=0)
        dg_query = compute_dg(x_sample, model)

        # Obtain the sample predicted label
        y_pred = predictions[i]

        # Compute similarity of the test point to the sampled points
        similarities[i] = aggregators[y_pred].similarity(dg_query)


    # Sort points by their similarity
    sorted_keys = sorted(similarities, key=similarities.get)
    sorted_vals = [x_samples[i] for i in sorted_keys]

    # Extract least similar 30 points
    sorted_vals = sorted_vals[:40]

    visualize_samples(sorted_vals)






def main():
    """
    Script demonstrating uncertainty functionality offered by dep. graphs.

    Sorts MNIST points by their uncertainty and prints them out.

    Idea: high uncertainty points are "strange" and seem atypical, compared to training data
    """

    # Load dataset
    train_x, train_y, test_x, test_y = load_mnist()

    # Split datasets by label into sub-datasets
    sub_datasets = {}

    for label in range(10):
        ds, _ = filter_dataset((train_x, train_y), [label])
        sub_datasets[label] = ds


    # Create model
    input_shape = train_x.shape[1:]
    model = build_sample_model(input_shape)

    # Train model
    model.fit(x=train_x, y=train_y, epochs=4)


    # Select points to inspect
    selected_points = test_x[:100]

    # Visualise points, sorted by their uncertainty
    sort_uncertain_points(selected_points, sub_datasets, model, n_samples=100)










main()