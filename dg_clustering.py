import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import pylab

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



def dendrogram_clustering(dg_sim_matrix, data):

    # Convert similarity matrix to condensed form
    cond_sim_matrix = squareform(dg_sim_matrix)

    # Compute hierarchical clusters
    Z = hierarchy.linkage(y=cond_sim_matrix, method="average")

    n_samples = data.shape[0]
    fig = pylab.figure(figsize=(50, 50))
    ax2 = fig.add_axes([0.0, 0.50, 1.0, 0.4])
    dendrogram = hierarchy.dendrogram(Z, labels=np.arange(n_samples))
    ax2.set_xticks([])
    ax2.set_yticks([])


    leaf_labels = dendrogram["ivl"]
    sorted_data = data[leaf_labels]

    cols = n_samples
    rows = 1

    for i in range(1, n_samples+1):
        sample = sorted_data[i - 1][:, :, 0]
        ax = fig.add_subplot(1, cols, i)
        ax.imshow(sample, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

    # TODO: how to plot images as the leaves?
