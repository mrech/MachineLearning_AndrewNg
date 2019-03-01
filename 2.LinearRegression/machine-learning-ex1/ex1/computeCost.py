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
    sqrErrors = (prediction.transpose()-y).transpose()**2

    return 1/(2*m) * np.sum(sqrErrors)
