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


'''
IMPLEMENTATION NOTE: When normalizing the features, it is important
to store the values used for normalization - the mean value and the stan-
dard deviation used for the computations. After learning the parameters
from the model, we often want to predict the prices of houses we have not
seen before. Given a new x value (living room area and number of bed-
rooms), we must first normalize x using the mean and standard deviation
that we had previously computed from the training set
'''
