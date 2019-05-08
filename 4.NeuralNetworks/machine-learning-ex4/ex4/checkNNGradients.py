
# CHECKNNGRADIENTS Creates a small neural network to check the
# backpropagation gradients


def checkNNGradients(lambda_param=0):
    '''
    CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
    backpropagation gradients, it will output the analytical gradients
    produced by your backprop code and the numerical gradients (computed
    using computeNumericalGradient). These two gradient computations should
    result in very similar values.
    '''

    from debugInitializeWeights import debugInitializeWeights
    import numpy as np
    from nnCostFunction import nnCostFunction
    from computeNumericalGradient import computeNumericalGradient

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # Generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to generate X
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = 1 + np.mod(np.arange(1, m+1), num_labels)

    # Unroll parameters
    nn_params = []
    nn_params.extend((list(Theta1.flatten()) +
                      list(Theta2.flatten())))

    # Short hand for cost function: ANONYMOUS FUNCTION
    def costFunc(p): 
          return nnCostFunction(p, input_layer_size, hidden_layer_size,
                                           num_labels, X, y, lambda_param)

    _, grad = costFunc(nn_params)

    numgrad = computeNumericalGradient(costFunc, nn_params)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar.
    print('The following two columns should be very similar.\n')
    print('\ngradApprox:\n {}'.format(numgrad))
    print('\ndeltaVector:\n{}'.format(np.array(grad)))

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

    print('\n\nIf your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          '\nRelative Difference: %g\n' % diff)


'''
Note:
ANONYMOUS FUNCTION
f = @(t)( 10*t )
f = lambda t: 10*t
'''
