import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import pylab


def dendrogram_clustering(dg_sim_matrix, data):

    # Convert similarity matrix to condensed form
    cond_sim_matrix = squareform(dg_sim_matrix)

    # Compute hierarchical clusters
    Z = hierarchy.linkage(y=cond_sim_matrix, method="average")

    # Specify position of dendrogram
    n_samples = data.shape[0]
    fig = pylab.figure(figsize=(100, 8))
    ax2 = fig.add_axes([0.0, 0.50, 1.0, 0.3])

    # Compute dendrogram
    dendrogram = hierarchy.dendrogram(Z, labels=np.arange(n_samples))
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Retrieve sorted images
    leaf_labels = dendrogram["ivl"]
    sorted_data = data[leaf_labels]

    # Plot images at the leaves
    cols = n_samples

    for i in range(1, n_samples+1):
        sample = sorted_data[i - 1][:, :, 0]
        ax = fig.add_subplot(1, cols, i)
        ax.imshow(sample, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()