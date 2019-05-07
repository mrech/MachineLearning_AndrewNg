
## SIGMOID Compute sigmoid functoon

def sigmoid(z):
    '''
    J = SIGMOID(z) computes the sigmoid of z.
    '''
    
    import numpy as np

    g = 1.0/(1.0 + np.exp(-z))

    return g