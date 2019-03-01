
# Plot the training data into a figure
import matplotlib.pyplot as plt


def plotData(x, y):
    '''
    Input: Requires x and y arguments
    Output: Plots the data points and gives the figure axes labels.
    '''
    plt.plot(x, y, 'rx', markersize=10)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')

    return plt.show()
