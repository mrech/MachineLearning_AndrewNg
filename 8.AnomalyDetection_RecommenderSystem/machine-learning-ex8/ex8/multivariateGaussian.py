def multivariateGaussian(X, mu, Sigma2):
    '''
    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability 
    density function of the examples X under the multivariate gaussian 
    distribution with parameters mu and Sigma2.
    '''

    import numpy as np

    k = len(mu)
    p = np.zeros((X.shape[0]))

    # Create covariance matrix
    cov = np.diag(Sigma2)

    x_m = X - mu.reshape((1, k))

    # multivariate Gaussian implementation > https://stackoverflow.com/a/26613290
    for i in range(len(p)):
        p[i] = (2*np.math.pi)**(-k/2)*np.linalg.det(cov)**(-1/2) * \
            np.exp(-1/2 *
                   np.dot(x_m.T[:, i].reshape((k, 1)).T, np.linalg.pinv(cov)) @
                   x_m.T[:, i])

    return p
