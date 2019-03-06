# Normal Equations
import numpy as np


def normalEqn(X, y):
    '''
    Requires X and y.
    Computes the closed-form solution to linear regression using Normal Equations.
    '''

    theta = np.zeros((X.shape[1], 1))
    theta = np.linalg.inv(X.transpose() @ X) @ (X.transpose() @ y)

    return theta
