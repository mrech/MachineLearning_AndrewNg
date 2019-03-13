# Compute cost and gradient for logistic regression
import math
import numpy as np
from sigmoid import *


def costFunction(theta, X, y):
    '''
    J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    parameter for logistic regression and the gradient of the cost
    w.r.t. to the parameters.
    '''

    (m, n) = X.shape
    theta = theta.reshape((n, 1))

    h = sigmoid(np.dot(X, theta))

    # VECTORIZE IMPLEMENTATION
    # log(0) in log(1-h) - floating-point error handling
    # you can solve this with feature scaling
    with np.errstate(divide='ignore'):
        J = 1/m*(-np.dot(np.transpose(np.vstack(y)), np.log(h))
                 - np.dot(np.transpose(np.vstack(1-y)), np.log(1-h)))

    grad = np.dot(np.transpose(X), (h-np.vstack(y)))/m

    return J, grad.flatten()
