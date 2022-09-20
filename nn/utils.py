
"""
Module that contains that contains a couple of utility functions
"""

import numpy as np
import os
import torch

def load(filename):

    """
    Loads the data that is provided
    @param filename: The name of the data file. Can be either 'tux_train.dat' or 'tux_val.dat'
    @return images: Numpy array of all images where the shape of each image will be W*H*3
    @return labels: Array of integer labels for each corresponding image in images
    """

    try:
        data = np.load(filename)
        data = torch.Tensor(data)
    except Exception as e:
        print('Check if the filepath of the dataset is {}'.format(os.path(filename)))
    print(data)
    images, labels = data[:,1:9], data[:,0:1]
    return images, labels


images, labels = load(os.path.join('datasets/test_data.npy'))
print(images)
print(labels)
