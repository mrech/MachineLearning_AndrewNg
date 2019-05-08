# SIGMOIDGRADIENT returns the gradient of the sigmoid function evaluated at z


def sigmoidGradient(z):
    '''
    g_prime = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
    evaluated at z. This should work regardless if z is a matrix or a
    vector. In particular, if z is a vector or matrix, you should return
    the gradient for each element.
    '''

    import numpy as np
    from sigmoid import sigmoid

    # Transform z into an array
    z = np.array(z)

    g_prime = np.zeros((z.shape))

    g_prime = np.dot(sigmoid(z), (1-sigmoid(z)))

    return g_prime
