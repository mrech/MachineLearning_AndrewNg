# PLOTDATA Plots the data points X and y into a new figure 

def plotData(X, y):
    '''
    PLOTDATA(x,y) plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 matrix.
    '''

    import matplotlib.pyplot as plt
    import numpy as np

    # Devide Positive and Negative Examples
    posIndex = np.where(y == 1)[0]
    negIndex = np.where(y == 0)[0]

    # Plot Examples
    plt.plot(X[posIndex,0], X[posIndex, 1], 'k+', X[negIndex,0], X[negIndex,1], 'yo')