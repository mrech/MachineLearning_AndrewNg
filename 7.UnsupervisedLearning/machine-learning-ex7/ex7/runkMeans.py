# RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
# is a single example

def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    '''
    [centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
    plot_progress) runs the K-Means algorithm on data matrix X, where each 
    row of X is a single example. It uses initial_centroids used as the
    initial centroids. max_iters specifies the total number of interactions 
    of K-Means to execute. plot_progress is a true/false flag that 
    indicates if the function should also plot its progress as the 
    learning happens. This is set to false by default. runkMeans returns 
    centroids, a Kxn matrix of the computed centroids and idx, a m x 1 
    vector of centroid assignments (i.e. each entry in range [1..K])
    '''

    import numpy as np
    from findClosestCentroids import findClosestCentroids 
    from plotProgresskMeans import plotProgresskMeans
    from computeCentroids import computeCentroids
    import matplotlib.pyplot as plt

    # Initialize values
    m, _ = np.shape(X)
    K = np.size(initial_centroids, 0)
    centroids = initial_centroids
    acc_centroids = centroids
    idx = np.zeros((m, 1))

    # Run K-Means

    for i in range(max_iters):

        # Output progress
        print('K-Means iteration %d/%d...\n' % (i, max_iters))

        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)

        # Optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, acc_centroids, idx, K, i)
            plt.show()

        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)
        acc_centroids = np.append(acc_centroids, centroids, axis=0)

    return centroids, idx




