# FEATURENORMALIZE Normalizes the features in X


def featureNormalize(X):
    '''
    FEATURENORMALIZE(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    '''

    import numpy as np

    mu = np.mean(X)
    sigma = np.std(X)
    X_norm = (X - mu)/sigma

    return X_norm, mu, sigma
