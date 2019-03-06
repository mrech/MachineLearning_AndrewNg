# Performs gradient descent to learn theta by taking
# num_iters gradient steps with learning rate alpha
import numpy as np
from computeCost import *


def gradientDescent(X, y, theta, alpha, num_iters):
    '''
    Input: X,y,theta and num_iters
    Output: updated theta and cost function history
    '''
    # Initialize some useful values
    m = len(y)  # number of training examples
    J_history = np.zeros((num_iters, 1))

    for iter in range(num_iters):

        hypothesis = X @ theta
        delta = 1/m * ((hypothesis.transpose()-y) @ X)

        theta = theta - (alpha * delta.transpose())

        # Save the cost J in every iteration
        J_history[iter] = computeCost(X, y, theta)

    return (theta, J_history)
