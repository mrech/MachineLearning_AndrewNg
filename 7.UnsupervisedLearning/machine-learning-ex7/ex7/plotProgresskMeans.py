# PLOTPROGRESSKMEANS is a helper function that displays the progress of 
# k-Means as it is running. It is intended for use only with 2D data.

def plotProgresskMeans(X, centroids, idx, K, i=0):
     '''
    PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
    points with colors assigned to each centroid. With the previous
    centroids, it also plots a line between the previous locations and
    current locations of the centroids.
     '''

     from plotDataPoints import plotDataPoints
     import matplotlib.pyplot as plt
     import numpy as np

     # Plot the examples
     plotDataPoints(X, idx, K)

     # Plot the centroids as black x's

     plt.plot(centroids[:,0], centroids[:, 1], 'x', \
         markeredgecolor = '#414042',  markersize = 7, markeredgewidth = 2)

     # Plot the history of the centroids with lines
     for j in range(K):
        # Group for centroids
         k = centroids[range(j, centroids.shape[0], K),:]
         plt.plot(k[:,0], k[:,1], color = 'k', linewidth = 0.5)

     # Title
     plt.title('Iteration number %d' % (i))


