# RBFKERNEL returns a radial basis function kernel between x1 and x2

def gaussianKernel(x1, x2, sigma):
    '''
    sim = gaussianKernel(x1, x2) returns the similarity between x1
    and x2 computed using a Gaussian kernel with bandwidth sigma
    '''

    import numpy as np

    sim = np.exp(-sum(np.power(x1-x2,2))/(2*np.power(sigma,2)))

    return sim