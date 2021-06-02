from tensorflow.keras.datasets import cifar10
from matplotlib.pyplot import imsave
import numpy as np
import os


def save_images(nbr, location='assets/images'):
    """
    Save some image of CIFAR-10 in the specified location.
    :param location: String
    :param nbr: Integer
        Number of image to store.
    :return: None
    """

    # Load the dataset.
    (x_train, y_train), (_, _) = cifar10.load_data()

    # Scale the pixel value to `-1 and 1` range.
    x_train = x_train.astype('float32') / 255.0
    classes = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck'])
    for i in range(nbr):
        imsave(f'{location}/{classes[y_train[i]]}_{i}.png', x_train[i])

