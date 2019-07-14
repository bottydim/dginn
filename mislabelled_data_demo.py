from mnist_loader import load_mnist, build_sample_model
import numpy as np
from dataset_utils import filter_dataset
from aggregator_utils import get_count_aggregators, uncertainty_pred, compute_dg


"""
Things to try:
    - Dependency graph computation approach (dg_relevance)
    - Aggregation approach + similarity to aggregator
    - Number of aggregator samples
    - Depth of the model
"""



def main():

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

    # Obtain subset of incorrectly labelled training points
    preds = np.argmax(model.predict(train_x), axis=1)
    incorrects = np.nonzero(preds != train_y)[0]
    incorrect_samples, incorrect_labels = train_x[incorrects], train_y[incorrects]
    incorrect_samples, incorrect_labels = incorrect_samples[:100], incorrect_labels[:100]

    # Select random set of samples
    n_mislabelled = 100
    indices = np.random.choice(train_x.shape[0], n_mislabelled, replace=False)
    random_x, random_y = train_x[indices], train_y[indices]

    # Combine two sets of samples
    combined_x, combined_y = np.concatenate((incorrect_samples, random_x), axis=0), \
                             np.concatenate((incorrect_labels, random_y), axis=0)

    print("No samples: ", combined_x.shape[0])

    # Get sample aggregators
    aggregators = get_count_aggregators(train_x, train_y, model, 100)

    sim_scores = {}
    agg_labels = {}

    for i, sample in enumerate(combined_x):

        sample = np.expand_dims(sample, axis=0)
        dg = compute_dg(sample, model)
        pred_label, sim_score = uncertainty_pred(dg, aggregators)

        agg_labels[i] = False
        sim_scores[i] = sim_score

        y_pred = np.argmax(model.predict(sample), axis=1)



        # TODO: remove after testing
        # Temporary printing information
        if pred_label == combined_y[i]:

            agg_labels[i] = True

            print("correct sample ", i)

            if pred_label != y_pred:
                print("disagreement ", i, " predicted: ", pred_label, " model: ", y_pred)


    # Sort samples by similarity
    sorted_keys = sorted(sim_scores, key=sim_scores.get)
    print(sorted_keys)

    labels = [agg_labels[i] for i in sorted_keys]
    scores = [sim_scores[i] for i in sorted_keys]
    print(labels)
    print(scores)


    # Count how many mislabelled samples are in the second half of the sorted keys
    # (i.e. have high similarity)
    n_incorrect = len(incorrect_labels)
    mislabelled = sum([1 for key in sorted_keys[n_incorrect:] if key >= n_incorrect])
    print("Mislabelled: ", mislabelled)

    return sorted_keys



main()




"""

Facts:
    1) High certainty corresponds to higher chance of a correct prediction
    2) Mislabelled samples have higher certainty than misclassified ones (though difference is not pronounced enough)
    3) Sometimes, uncertainty gives correct estimate, whilst the model does not    

"""