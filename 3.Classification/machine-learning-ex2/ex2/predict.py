
# PREDICT Predict whether the label is 0 or 1 using learned logistic
# regression parameters theta
import numpy as np
from sigmoid import *


def predict(theta, X):
    '''
    p = PREDICT(theta, X) computes the predictions for X using a 
    threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    '''
    # Number of training examples
    (m, n) = X.shape

    # Initialize prediction
    p = np.zeros((m, 1))

    theta = theta.reshape((n, 1))

    # prediction
    p = sigmoid(np.dot(X, theta))

    # assign to the right class based on the defined threshold using list comprehension
    return [1 if i >= 0.5 else 0 for i in p]
