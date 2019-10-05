import matplotlib.pyplot as plt
import math


def visualize_samples(samples, similaries, title="", ):
    cols = 4
    rows = math.ceil(len(samples) / cols)
    fig = plt.figure(figsize=(8, 8))

    for i in range(1, len(samples) + 1):
        # Remove channel dimension
        sample = samples[i - 1][:, :, 0]

        # Plot image
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(sample, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("{}-sim-{}".format(i, round(similaries[i - 1], 3)))
        plt.tight_layout()
    fig.suptitle(title)
    return fig


def visualize_samples_informetis(samples, similaries, title="", figsize=(15, 15), show=True):
    from loadData import visualise_sample as visualise_time_series
    cols = 4
    rows = math.ceil(len(samples) / cols)
    fig = plt.figure(figsize=figsize)

    for i in range(1, len(samples) + 1):
        # Remove channel dimension
        y_axis_ = samples[i - 1, ...]

        # Plot image
        ax = fig.add_subplot(rows, cols, i)
        visualise_time_series(y_axis_, title="{}-sim-{}".format(i, round(similaries[i - 1], 3)), ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
    fig.suptitle(title)
    if show:
        plt.show(block=False)
    return fig
