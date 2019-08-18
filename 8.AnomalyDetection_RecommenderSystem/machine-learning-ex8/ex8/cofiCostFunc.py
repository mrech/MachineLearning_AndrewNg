# COFICOSTFUNC Collaborative filtering cost function


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_par):
    '''
    J, grad = COFICOSTFUNC(params, Y, R, num_users, num_movies,
    num_features, lambda) returns the cost and gradient for the
    collaborative filtering problem.
    '''
    import numpy as np

    # Unfold the U and W matrices from params
    X = np.reshape(params[:num_movies*num_features],
                   (num_movies, num_features), order='F')

    Theta = np.reshape(params[num_movies*num_features:],
                       (num_users, num_features), order='F')

    # You need to return the following values correctly
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    # Notes:
    #    X - num_movies x num_features matrix of movie features
    #    Theta - num_users x num_features matrix of user features
    #    Y - num_movies x num_users matrix of user ratings of movies
    #    R - num_movies x num_users matrix, where R(i, j) = 1 if the
    #        i-th movie was rated by the j-th user
    #    X_grad - num_movies x num_features matrix, containing the
    #             partial derivatives w.r.t. to each element of X
    #    Theta_grad - num_users x num_features matrix, containing the
    #                 partial derivatives w.r.t. to each element of Theta

    # use element-wise multiplication with R matrix to set entries to 0
    J = np.sum((np.dot(X, Theta.T)*R-Y*R)**2)/2 + \
        (lambda_par/2 * np.sum(np.power(Theta, 2))) +\
        (lambda_par/2 * np.sum(np.power(X, 2)))

    X_grad = np.dot(np.dot(X, Theta.T)*R-Y*R, Theta) + lambda_par*X
    Theta_grad = np.dot((np.dot(X, Theta.T)*R-Y*R).T, X) + lambda_par*Theta

    grad = []
    grad.extend((list(X_grad.flatten(order='F')) +
                 list(Theta_grad.flatten(order='F'))))

    # bad operand type for unary -: 'list'
    grad = np.array(grad)
    
    return J, grad


'''            
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
'''
