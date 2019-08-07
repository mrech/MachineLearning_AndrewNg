# FINDCLOSESTCENTROIDS computes the centroid memberships for every example

def findClosestCentroids(X, centroids):
    '''
    idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
    in idx for a dataset X where each row is a single example. idx = m x 1 
    vector of centroid assignments (i.e. each entry in range [1..K])
    '''

    import numpy as np

    # You need to return the following variables correctly.
    idx = np.zeros((np.size(X,0), 1))

    for m in range(np.size(X,0)):
        idx[m] = np.argmin(np.sum(((X[m] - centroids)**2), axis = 1), axis = 0)

    return idx

