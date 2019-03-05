# COMPUTECOST Compute cost for linear regression
# check the convergence of your gradient descent implementation
import numpy as np


def computeCost(X, y, theta):
    '''
    Input: X, y and theta
    Compute the cost of a particular choice of theta to fit X and y
    '''

    m = len(y)  # number of training examples
    prediction = X @ theta
    # successfully subtracting a vector from a matrix
    # vectorized form of the cost function
    sqrErrors = (prediction.transpose() -
                 y) @ (prediction.transpose()-y).transpose()

    return 1/(2*m) * sqrErrors
