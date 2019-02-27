import numpy as np

# Design matrix
X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.arange(1, 4).reshape(3, 1)
theta = np.array([[0], [1]])


def CostFunctionJ(X, y, theta):
    '''
    requires: X 'design matrix' of training examples, 
              y the class labels,
              theta the parameter vecor
    returns: mean squares errors 
    '''
    (m, n) = np.shape(X)
    predictions = X @ theta
    sqrErrors = (predictions-y)**2

    return 1/(2*m) * np.sum(sqrErrors)


CostFunctionJ(X, y, theta)

theta = np.array([[0],[0]])
CostFunctionJ(X, y, theta)