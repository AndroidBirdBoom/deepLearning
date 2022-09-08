import numpy as np
import matplotlib.pyplot as plt
import h5py
from cnn_utils import *

np.random.seed(1)

if __name__ == "__main__":
    train_data = h5py.File("datasets/train_signs.h5", 'r')
    test_data = h5py.File("datasets/test_signs.h5", 'r')
    print('train_data', train_data.keys(), "test_data", test_data.keys())
    plt.rcParams['figure.figsize'] = (5.0, 4.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

