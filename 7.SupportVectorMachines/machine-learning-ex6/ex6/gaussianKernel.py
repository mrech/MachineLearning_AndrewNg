# RBFKERNEL returns a radial basis function kernel between x1 and x2

def gaussianKernel(x1, x2, sigma=0.1):
    '''
    sim = gaussianKernel(x1, x2) returns the 
    similarity between x1 and x2 computed using a Gaussian 
    kernel with bandwidth sigma
    '''

    import numpy as np

    # Ensure that x1 and x2 are column vectors
    x1 = x1.flatten()
    x2 = x2.flatten()

    sim = np.exp(-np.sum(np.power((x1-x2),2))/(2*(sigma**2)))

    return sim