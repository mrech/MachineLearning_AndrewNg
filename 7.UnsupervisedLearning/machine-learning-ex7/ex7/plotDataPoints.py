# PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
# index assignments in idx have the same color

def plotDataPoints(X, idx, K):
    '''
    PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those 
    with the same index assignments in idx have the same color
    '''

    import matplotlib.pyplot as plt

    # Create palette
    colors = idx.flatten()

    # Plot the data
    plt.scatter(X[:,0], X[:,1], c = colors, cmap='prism')