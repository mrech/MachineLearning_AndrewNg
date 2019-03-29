# Compute cost and gradient for logistic regression with 
# regularization


def lrCostFunction(theta, X, y, RegParam):
    '''
    J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters. 
    '''
    from sigmoid import sigmoid
    import numpy as np

    (m, n) = X.shape
    theta = theta.reshape((n, 1))
    h = sigmoid(np.dot(X, theta))

    # Regularize Cost function:
    # NB. Regularization term starts at theta1
    J = -1/m * (np.dot(np.transpose(np.vstack(y)), np.log(h))
                + np.dot(np.transpose(np.vstack(1-y)), np.log(1-h))) \
        + (RegParam/(2*m)) * np.sum(np.power(theta[1:n], 2))

    # NB. Regularization term starts at theta1
    # compute regularization term for all
    grad = np.dot(np.transpose(X), (h-np.vstack(y)))/m \
        + np.dot((RegParam/m), theta)

    # adjust for the fist term, theta0
    grad[0] = np.dot(np.transpose(np.vstack(X[:, 0])), (h-np.vstack(y)))/m


    return J, grad.flatten()


'''



% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)

'''