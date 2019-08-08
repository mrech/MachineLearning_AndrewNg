# KMEANSINITCENTROIDS This function initializes K centroids that are to be
# used in K-Means on the dataset X


def kMeansInitCentroids(X, K):
    '''
    centroids = KMEANSINITCENTROIDS(X, K) returns K initial randomly chosen 
    centroids to be used with the K-Means on the dataset X
    '''

    import numpy as np

    # You should return this values correctly
    centroids = np.zeros((K, np.size(X, 1)))

    # Initialize the centroids to be random examples
    # Randomly reordedr the indices of examples
    randidx = np.random.permutation(np.size(X, 0))

    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]

    return centroids
