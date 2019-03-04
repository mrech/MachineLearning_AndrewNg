# Feature Normalization:
# good preprocessing step when working with learning algorithms.
import numpy as np
import math


def featureNormalize(X):
    '''
    Requires X features to normalize. 
    Returns a normalized version of X where the mean value of each
    feature is 0 and the standard deviation is 1.
    '''

    X_norm = X[:]  # Clone
    mu = np.mean(X)
    sigma = np.std(X)

    for i in range(X.shape[1]):
        X_norm[i] = (X[i] - mu[i])/sigma[i]

    return X_norm, mu, sigma
