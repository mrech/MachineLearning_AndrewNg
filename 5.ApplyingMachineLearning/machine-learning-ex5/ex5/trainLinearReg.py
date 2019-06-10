# TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
# regularization parameter lambda


def trainLinearReg(X, y, lambda_par):
    '''
    [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
    the dataset (X, y) and regularization parameter lambda. Returns the
    trained parameters theta.
    '''
    import numpy as np
    from linearRegCostFunction import linearRegCostFunction
    from scipy import optimize


    # Initialize Theta
    initial_theta = np.zeros((X.shape[1]))

    # Create "short hand" for the cost function to be minimized
    def costFunction(t): return linearRegCostFunction(X, y, t, lambda_par)

    # Now, costFunction is a function that takes in only one argument

    # Minimize using optimize.minimize
    result = optimize.minimize(costFunction, initial_theta,
                                    method='CG', jac=True,
                                    options={'maxiter': 200})
    return result.x
