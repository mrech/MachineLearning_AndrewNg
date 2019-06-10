# LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
# regression with multiple variables


def linearRegCostFunction(X, y, theta, lambda_par):
    '''
    [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda_par) computes the 
    cost of using theta as the parameter for linear regression to fit the 
    data points in X and y. Returns the cost in J and the gradient in grad
    '''

    import numpy as np

    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # Calculate predicted values
    h = X @ theta

    # Regularized linear regression cost function
    # NB. Regularization term starts at theta1
    J = (1/(2*m)) * np.sum(np.power(h-y, 2)) \
        + (lambda_par/(2*m)) * np.sum(np.power(theta[1:m], 2))

    # Compute regularization term for all
    grad = np.dot(np.transpose(X), h - y)/m + \
        np.dot((lambda_par/m), theta)

    # Adjust for the first term, theta0
    grad[0] = np.dot(np.transpose(X[:,0]), h - y)/m

    return [J, grad]
