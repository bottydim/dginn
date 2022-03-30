from mnist_loader import load_mnist
from dataset_utils import filter_dataset
import imageio
from demo.data_loaders.mnist_loaders import randomly_sample


def save_mnist_imgs(folder_path):

    _, _, test_x, test_y = load_mnist()

    n_samples = 300
    ones_x, ones_y = filter_dataset((test_x, test_y), [1])
    ones_x, ones_y = randomly_sample(ones_x, ones_y, n_samples)

    for i in range(n_samples):
        f_path = folder_path + str(i) + ".jpg"
        next_one = ones_x[i]
        imageio.imwrite(f_path, next_one)




folder_path = "/Users/AdminDK/Desktop/mnist_imgs/"
save_mnist_imgs(folder_path)