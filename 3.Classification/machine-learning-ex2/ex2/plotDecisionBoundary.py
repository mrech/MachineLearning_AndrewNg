# Plots the data points X and y into a new figure with
# the decision boundary defined by theta
from plotData import *
from mapFeature import *
import numpy as np
import matplotlib.pyplot as plt


def plotDecisionBoundary(theta, X, y):
    '''
    PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
    positive examples and o for the negative examples. X is assumed to be 
    a either 
    1) Mx3 matrix, where the first column is an all-ones column for the 
       intercept.
    2) MxN, N>3 matrix, where the first column is all-ones
    '''
    # Plot Data
    plotData(X.iloc[:, 1:3], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[1])-2,  max(X[1])+2])

        # Calculate the decision boundary line: 
        # g(z) = 1/2 >> e^(-z) = 1 >> z = 0 
        # theta0 + theta1X1 + theta2X2 = 0
        # x2 plays as y >> y = - (theta0 + theta1X1) / theta2
        plot_y = (-1/theta[2])*(theta[0]+(theta[1]*plot_x))

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y, label='Decision Boundary')
        plt.ylim(bottom=30, top=100)
        plt.xlim(left=30, right=100)

        # Legend, specific for the exercise
        plt.legend()

    # plots non-linear decision boundary
    else:
        # Create an evenly spaced grid (0 >= predictions <= 1)
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        # Classifier predictions: initiate
        z = np.zeros((len(u), len(v)))
        # Evaluate z = theta*x over the grid
        theta = theta.reshape((theta.shape[0], 1))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = np.dot(mapFeature(u[i], v[j]), theta)
        # important to transpose z before calling contour
        z = np.transpose(z)

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        CS = plt.contour(u, v, z, 0, colors='b', linewidths=2)
        labels = ['Decision Boundary']
        for i in range(len(labels)):
            CS.collections[i].set_label(labels[i])
