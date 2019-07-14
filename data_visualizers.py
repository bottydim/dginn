import matplotlib.pyplot as plt
import math


def visualize_samples(samples):

    cols = 4
    rows = math.ceil(len(samples)/cols)
    fig = plt.figure(figsize=(8, 8))

    for i in range(1, len(samples)+1):

        # Remove channel dimension
        sample = samples[i-1][:, :, 0]

        # Plot image
        fig.add_subplot(rows, cols, i)
        plt.imshow(sample, cmap='gray')

    plt.show()
