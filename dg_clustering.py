import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


def get_hierarchical_clustering(dg_sim_matrix):
    """
    :param dg_sim_matrix: an [n_samples, n_samples] numpy matrix, storing pairwise
                          similarity of input dependency graphs

    returns:
    """
    clustering = AgglomerativeClustering(n_clusters=10,
                                         affinity="precomputed",
                                         linkage="average")

    clustering.fit(dg_sim_matrix)

    # Get cluster labels for each point
    label = clustering.labels_



def dendrogram_clustering(dg_sim_matrix, labels):

    # Convert similarity matrix to condensed form
    cond_sim_matrix = squareform(dg_sim_matrix)

    # Compute hierarchical clusters
    Z = hierarchy.linkage(y=cond_sim_matrix, method="average")

    dendrogram = hierarchy.dendrogram(Z, labels=labels)

    plt.show()

    # TODO: how to plot images as the leaves?
