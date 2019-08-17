# CHECKCOSTFUNCTION Creates a collaborative filering problem
# to check your cost function and gradients


import numpy as np


def checkCostFunction(lambda_par=0):
    '''
    CHECKCOSTFUNCTION(lambda) Creates a collaborative filering problem 
    to check your cost function and gradients, it will output the 
    analytical gradients produced by your code and the numerical gradients 
    (computed using computeNumericalGradient). These two gradient 
    computations should result in very similar values.
    '''

    import numpy as np
    from cofiCostFunc import cofiCostFunc
    from computeNumericalGradient import computeNumericalGradient

    # Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.rand(np.shape(Y)[0], np.shape(Y)[1]) > 0.5] = 0
    R = np.zeros(np.shape(Y))
    R[Y != 0] = 1

    # Run Gradient Checking
    X = np.random.randn(np.shape(X_t)[0], np.shape(X_t)[1])
    Theta = np.random.randn(np.shape(Theta_t)[0], np.shape(Theta_t)[1])
    num_users = np.shape(Y)[1]
    num_movies = np.shape(Y)[0]
    num_features = Theta_t.shape[1]

    # Unroll parameters
    params = []
    params.extend((list(X.flatten(order='F')) +
                   list(Theta.flatten(order='F'))))

    # Short hand for cost function:
    def costFunc(t):
        return cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lambda_par)

    numgrad = computeNumericalGradient(costFunc, params)

    _, grad = costFunc(params)

    print('The above two columns you get should be very similar.\n')
    print('\n Your Numerical Gradient:\n {}'.format(numgrad))
    print('\nAnalytical Gradient:\n{}'.format(np.array(grad)))

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('\nIf your cost function implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          '\nRelative Difference: %g\n' % (diff))
