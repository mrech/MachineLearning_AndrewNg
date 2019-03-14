# Compute cost and gradient for logistic regression with regularization
from sigmoid import *


def costFunctionReg(theta, X, y, RegParam):
    '''
    J = COSTFUNCTIONREG(theta, X, y, RegParam) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters. 
    '''
    (m, n) = X.shape
    theta = theta.reshape((n, 1))
    h = sigmoid(np.dot(X, theta))

    # Regularize Cost function:
    # NB. Regularization term starts at theta1
    J = 1/m * (-np.dot(np.transpose(np.vstack(y)), np.log(h))
               - np.dot(np.transpose(np.vstack(1-y)), np.log(1-h))) \
        + (RegParam/(2*m)) * np.sum(np.power(theta[1:n], 2))

    # NB. Regularization term starts at theta1
    # compute regularization term for all
    grad = np.dot(np.transpose(X), (h-np.vstack(y)))/m \
        + np.dot((RegParam/m), theta)

    # adjust for the fist term, theta0
    grad[0] = np.dot(np.transpose(np.vstack(X.iloc[:, 0])), (h-np.vstack(y)))/m

    return J, grad.flatten()
