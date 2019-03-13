# Plots the data points X and y into a new figure
import matplotlib.pyplot as plt


def plotData(X, y):
    '''
    X is assumed to be a Mx2 matrix.
    plots the data points with + for the positive examples
    and o for the negative examples
    '''

    # Find indices of positive and negative examples
    pos = y.index[y == 1]
    neg = y.index[y == 0]

    # Plot Examples
    plt.plot(X.iloc[pos, 0], X.iloc[pos, 1], 'k+', label='Admitted',
             linewidth=2, markersize=7)

    plt.plot(X.iloc[neg, 0], X.iloc[neg, 1], 'yo', label='Not admitted',
             linewidth=2, markersize=7)

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.title('Scatter plot of training data')
    plt.legend()
