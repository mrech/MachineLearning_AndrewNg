# SIGMOID Compute sigmoid function

def sigmoid(x):
    '''
    sigmoid(z) compustes the sigmoid of z
    '''
    import numpy as np
    
    try:
        g = np.zeros(x.shape)
    except AttributeError:  # 'int' object has no attribute 'shape'
        g = 0

    g = 1/(1 + np.exp(-x))

    return g
