from loadData import load_informetis
from demo.uncertainty_demo import save_fig

if __name__ == '__main__':
    import tensorflow as tf

    tf.enable_eager_execution()

from aggregator_utils import extract_dgs_by_ids, \
    get_aggregators_from_collection
from core import *
import os
from aggregator_utils import compute_dg_per_datapoint
from dginn.core import Activations_Computer
from dginn.aggregator_utils import get_aggregators_from_collection


def informetis(n_samples=1000):
    from loadData import get_model, get_data_points_paths, get_all_dp_per_class, get_human_labels, get_labels_str
    root_folder = '../'
    # load model
    model = get_model()
    data_folder = '../' + root_folder
    data_points = get_data_points_paths(os.path.join(data_folder, "data/botty/"))
    # get 100 data-points for each class
    data_set_cls_dict = get_all_dp_per_class(data_points, n_samples, threshold=1, pos_threshold=50)

    # create dictionaries for different forms of labels
    class_names = [cls for cls in data_set_cls_dict.keys()]
    label_dict = get_human_labels(path=os.path.join(root_folder, "initial_data.json"))
    # list of string values of data_set_cls_dict
    labels_str = get_labels_str(label_dict)
    # the y-index of the classes in their current order
    current_labels = [labels_str.index(cls) for cls in data_set_cls_dict.keys()]
    class_names_human = [label_dict[labels_str[lbl]] for lbl in current_labels]
    # turn data_set_cls_dict into an array
    cls_data_sets = [data_set_cls_dict[cls] for cls in data_set_cls_dict.keys()]


    ###################################
    # EXPERIMENTS
    ###################################

    # EXPERIMENT #1
    visualise_kernels(model)
    # EXPERIMENT #2
    analyze_dependence(model, l_num=2)

    # EXPERIMENT #3
    # select first class
    cls_id = 0
    dependency_graph_showcase(model, cls_data_sets, cls_id)
    # EXPERIMENT #4
    prototypical_examples(class_names_human, cls_data_sets, model)


def dependency_graph_showcase(model, cls_data_sets, cls_id, label_dict, class_names):
    from dg_relevance import compute_activations_gen, compute_weight, compute_weight_activations_gen, \
        compute_grads_gen, relevance_select, \
        combine_compute_fx, view_input_layers, view_layer_importance, layer_importance_strategies, get_input_layers, \
        cf_dgs_intersect, cf_dgs_union, visualise_layers_DG

    data_set = cls_data_sets[cls_id]

    #############################################################
    #### SHOWCASE Relevance Computation ############
    # compute relevances using different strategies
    compute_fx = compute_activations_gen(data_set, fx_modulate=np.abs, verbose=False)
    relevances_a = compute_fx(model)
    relevances_w = compute_weight(model, fx_modulate=np.abs)
    compute_fx = compute_weight_activations_gen(data_set, fx_modulate=np.abs, verbose=False)
    relevances_wa = compute_fx(model)
    compute_fx = compute_grads_gen(data_set, fx_modulate=np.abs, verbose=False)
    relevances_g = compute_fx(model)
    relevances_ga = combine_compute_fx(relevances_a, relevances_g)
    relevances_wa_app = combine_compute_fx(relevances_a, relevances_w)
    relevances_ga_app = combine_compute_fx(relevances_a, relevances_g)
    #############################################################

    # visualise layer importance
    input_layers = get_input_layers(model, include_adjacent=True)
    view_input_layers(relevances_a, input_layers)
    #
    view_layer_importance(relevances_a, model)
    view_layer_importance(relevances_g, model, summary_fx=np.std)
    relevance_strategies = [relevances_w, relevances_a, relevances_wa, relevances_g, relevances_ga]
    relevance_names = ["Weights", "Activations", "Weights * Activations", "Grads", "Grads * Activations"]
    layer_importance_strategies(relevance_strategies, relevance_names, logy=True)

    #############################################################
    # compute importance values using each importance computation strategy
    relevance_list_act = compute_class_relevance(model, cls_data_sets, compute_activations_gen)
    relevance_list_grads = compute_class_relevance(model, cls_data_sets, compute_grads_gen)
    relevance_list_wa = compute_class_relevance(model, cls_data_sets, compute_weight_activations_gen)
    relevance_list_act += [1]

    #############################################################
    # set relevance strategy to use!
    relevance_list_cf = relevance_list_grads
    #############################################################

    # compute dependency graph for each class
    DG_cls = []
    for r in relevance_list_cf:
        DG_cls.append(relevance_select(r, input_layer=input_layers, threshold=0.2))

    # get all neurons for all layers
    DG_full = relevance_list_wa[1]
    # compute intersections between dependency graphs
    DG_inter = cf_dgs_intersect(DG_cls)
    DG_union = cf_dgs_union(DG_cls)

    # visualise cluster_heatmap
    cluster_heatmap(DG_cls, class_names, label_dict)

    # visualised shared neurons (intersection)
    visualise_layers_DG(DG_inter, DG_full, model, percentage=True, )

    # visual shared neurons (union)
    visualise_layers_DG(DG_union, DG_full, model, percentage=True, )


def cluster_heatmap(DG_cls, class_names, label_dict):
    from dg_relevance import cf_dgs_unique
    from sklearn.preprocessing import normalize
    import seaborn as sns
    num_cls = len(DG_cls)
    matrix_unique_pair = np.zeros((num_cls, num_cls))
    for i, dg_a in enumerate(DG_cls):
        for j, dg_b in enumerate(DG_cls):
            matrix_unique_pair[i, j] = cf_dgs_unique(dg_a, dg_b)
    matrix_unique_pair = normalize(matrix_unique_pair, axis=1, norm='max')
    class_names_human = [label_dict[lbl] for lbl in class_names]
    cg = sns.clustermap(matrix_unique_pair,
                        xticklabels=class_names_human, yticklabels=class_names_human,
                        metric="cityblock",
                        method="single"
                        )
    axes = plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)


def compute_class_relevance(model, data_sets, compute_fx_gen):
    relevance_list_act = []
    for i, ds in enumerate(data_sets):
        #     print("Data set #{}".format(i))
        X_temp = ds
        compute_fx = compute_fx_gen(X_temp, fx_modulate=np.abs)
        relevance = compute_fx(model)
        relevance_list_act.append(relevance)
    return relevance_list_act


def analyze_dependence(model, l_num=2):
    from dg_relevance import visualise_kernel_dependence, compute_dependency_weight
    relevances = compute_dependency_weight(model)
    l = model.layers[l_num]
    importance_kernels_1 = relevances[l]
    visualise_kernel_dependence(importance_kernels_1, sharey=False, log=False)


def visualise_kernels(model):
    from dginn.dg_relevance import compute_weight_per_channel
    relevance_channels = compute_weight_per_channel(model)
    # visualise the efect of the kernels in the first layer
    importance = relevance_channels[model.layers[1]].numpy()
    n_rows = 2
    n_cols = int(int(importance.shape[-1]) / n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5), sharex="all", sharey="all")
    for i in range(n_rows):
        for j in range(n_cols):
            idx = 10 * i + j
            # kernel_size, input_dim, kernel_id
            y_axis_ = importance[:, 0, idx]
            line = axes[i, j].plot(np.arange(y_axis_.shape[0]), y_axis_)
            axes[i, j].set_title("Kernel #:{}".format(idx))
    plt.show()


def prototypical_examples(class_names_human, cls_data_sets, model):
    # dist lists
    dg_collection_list = []
    for i in range(len(cls_data_sets)):
        print("Dataset #{}".format(i))
        dg_collection = compute_dg_per_datapoint(cls_data_sets[i], model, Activations_Computer)
        dg_collection_list.append(dg_collection)
    aggregators = get_aggregators_from_collection(dg_collection_list)
    for cls in range(len(cls_data_sets)):
        print(class_names_human[cls])
        aggregator = aggregators[cls]
        dg_collection_query = dg_collection_list[cls]
        dataset = cls_data_sets[cls]
        _, _, [fig_most, fig_least] = informetis_prototypical(dataset, dg_collection_query,
                                                              aggregator, False)
        save_fig(cls, fig_most, "most", "informetis")
        save_fig(cls, fig_least, "least", "informetis")


def informetis_prototypical(dataset, dg_collection_query, aggregator, show):
    from dginn.aggregator_utils import get_number_datapoints, extract_dgs_by_ids

    similarities = {}
    num_datapoints = get_number_datapoints(dg_collection_query)

    for i in range(num_datapoints):
        dg_query = extract_dgs_by_ids(dg_collection_query, [i])

        # Compute similarity of the test point to the sampled points
        similarities[i] = aggregator.similarity(dg_query)

    # Sort points by their similarity
    sorted_keys = sorted(similarities, key=similarities.get, reverse=True)

    from data_visualizers import visualize_samples_informetis
    samples = dataset["aggPower"][sorted_keys]
    similarity_list = [similarities.get(key) for key in sorted_keys]

    similarity_list = [similarities.get(key) for key in sorted_keys]
    lim_samples = 10
    fig_most = visualize_samples_informetis(samples[:lim_samples, ...], similarity_list[:lim_samples], figsize=(20, 15),
                                            show=show)

    samples_rev = samples[::-1]
    fig_least = visualize_samples_informetis(samples_rev[:lim_samples, ...], similarity_list[::-1][:lim_samples],
                                             show=show)
    return samples, similarity_list, [fig_most, fig_least]


def informetis_one_cls(cls=0):
    cls_datasets, model = load_informetis()

    # dist lists
    dg_collection_list = []
    for i in range(len(cls_datasets)):
        print("Dataset #{}".format(i))
        dg_collection = compute_dg_per_datapoint(cls_datasets[i], model, Activations_Computer)
        dg_collection_list.append(dg_collection)

    from dginn.aggregator_utils import get_aggregators_from_collection, get_number_datapoints

    aggregators = get_aggregators_from_collection(dg_collection_list)
    dg_collection_query = dg_collection_list[cls]
    similarities = {}
    num_datapoints = get_number_datapoints(dg_collection_query)
    for i in range(num_datapoints):
        dg_query = extract_dgs_by_ids(dg_collection_query, [i])

        # Compute similarity of the test point to the sampled points
        similarities[i] = aggregators[cls].similarity(dg_query)

    # Sort points by their similarity
    sorted_keys = sorted(similarities, key=similarities.get, reverse=True)

    from data_visualizers import visualize_samples_informetis
    samples = cls_datasets[cls]["aggPower"][sorted_keys]
    visualize_samples_informetis(samples, similarities, title="Most Similar")
