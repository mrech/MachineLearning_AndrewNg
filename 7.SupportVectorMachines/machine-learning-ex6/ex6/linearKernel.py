# LINEARKERNEL returns a linear kernel between x1 and x2

def linearKernel(x1, x2):
    '''
    linearKernel(x1, x2) returns a linear kernel function

    '''

    import numpy as np

    return np.dot(x1, x2.T)
