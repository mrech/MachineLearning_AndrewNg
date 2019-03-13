# Plots the data points X and y into a new figure with
# the decision boundary defined by theta
from plotData import *
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

        # Calculate the decision boundary line
        plot_y = (-1/theta[2])*(theta[0]+(theta[1]*plot_x))

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y, label='Decision Boundary')
        plt.ylim(bottom=30, top=100)
        plt.xlim(left=30, right=100)

        # Legend, specific for the exercise
        plt.legend()


'''

else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)

'''
