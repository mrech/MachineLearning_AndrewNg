# LEARNINGCURVE Generates the train and cross validation set errors needed
# to plot a learning curve


def learningCurve(X, y, Xval, yval, lambda_par):
    '''
    LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
    cross validation set errors for a learning curve. In particular, 
    it returns two vectors of the same length - error_train and 
    error_val. Then, error_train(i) contains the training error for
    i examples (and similarly for error_val(i)).
    '''
    import numpy as np
    from trainLinearReg import trainLinearReg
    from linearRegCostFunction import linearRegCostFunction

    # In this function, you will compute the train and test errors for
    # dataset sizes from 1 up to m. In practice, when working with larger
    # datasets, you might want to do this in larger intervals.

    # Number of training examples
    m = X.shape[0]

    # You need to return these values correctly
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    # Loop over the examples with the following:
    for i in range(1, m + 1):
        # Use the trainLinearReg to find the theta parameters
        theta = trainLinearReg(X[0:i,:], y[0:i,:], lambda_par)
        # Compute train/cross validation errors
        J_train, _ = linearRegCostFunction(X[0:i,:], y[0:i,:], theta, 0)
        J_CV, _ = linearRegCostFunction(Xval, yval, theta, 0)
        # store the results
        error_train[i-1] = J_train
        error_val[i-1] = J_CV

    return error_train, error_val


'''  
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
'''
