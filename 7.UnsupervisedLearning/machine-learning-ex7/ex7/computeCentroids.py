# COMPUTECENTROIDS returns the new centroids by computing the means of the
# data points assigned to each centroid.


def computeCentroids(X, idx, K):
    '''
    centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
    computing the means of the data points assigned to each centroid. It is
    given a dataset X where each row is a single data point, a vector
    idx of centroid assignments (i.e. each entry in range [1..K]) for each
    example, and K, the number of centroids. You should return a matrix
    centroids, where each row of centroids is the mean of the data points
    assigned to it.
    '''

    import numpy as np

    # Useful variables
    _, n = np.shape(X)

    # You need to return the following variables correctly.
    centroids = np.zeros((K, n))

    for k in range(K):
        # extract the indeces of all examples within the specific centroid
        ideces = np.where(idx == k)[0]
        # extract exaples within the centroid and calculate the mean by rows
        centroids[k] = np.mean(X[ideces, :], axis=0)

    return centroids
