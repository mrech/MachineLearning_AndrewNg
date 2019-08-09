# PCA Run principal component analysis on the dataset X


def pca(X):
    '''  
    [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
    Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    '''

    import numpy as np

    # Useful values
    m, n = np.shape(X)

    # You need to return the following variables correctly.
    U = np.zeros(n)
    S = np.zeros(n)

    # first compute the covariance matrix
    Sigma = np.dot(X.transpose(), X)/m

    U, S, _ = np.linalg.svd(Sigma)

    return U, S