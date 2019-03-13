# SIGMOID Compute sigmoid function
import numpy as np


def sigmoid(x):
    '''
    sigmoid(z) compustes the sigmoid of z
    '''

    try:
        g = np.zeros(x.shape)
    except AttributeError:  # 'int' object has no attribute 'shape'
        g = 0

    g = 1/(1 + np.exp(-x))

    return g
