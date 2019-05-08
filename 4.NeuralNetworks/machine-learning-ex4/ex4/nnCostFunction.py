# NNCOSTFUNCTION Implements the neural network cost function for a two layer
# neural network which performs classification


def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_param):
    '''
    [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels,
    X, y, lambda) computes the cost and gradient of the neural network. 
    The parameters for the neural network are "unrolled" into the vector
    nn_params and need to be converted back into the weight matrices. 

    The returned parameter grad should be a "unrolled" vector of the
    partial derivatives of the neural network.
    '''

    import numpy as np
    from sigmoid import sigmoid
    from sigmoidGradient import sigmoidGradient

    # Reshape nn_params back into the parameters Theta1 and Theta2
    # the weight matrices for our 2 layer neural network

    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size*(input_layer_size+1))::],
                        (num_labels, (hidden_layer_size + 1)))

    # Setup some useful variables
    m = X.shape[0]

    # Retrun the following variables correctly
    J = []
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # Part 1:
    # Feedforward the neural network and return the cost in the variable J.

    for i in range(m):
        act_1 = X[i]
        act_1 = np.append(1, act_1)  # add 1
        z_2 = np.dot(Theta1, act_1)
        act_2 = sigmoid(z_2)
        act_2 = np.append(1, act_2)  # add 1
        z_3 = np.dot(Theta2, act_2)
        h = sigmoid(z_3)

        # Logical arrays (binary vector of 1's and 0's)
        y_vect = np.zeros(num_labels)
        y_vect[y[i]-1] = 1

        cost = -1/m * (np.dot(np.transpose(np.vstack(y_vect)), np.log(h)) +
                       np.dot(np.transpose(np.vstack(1-y_vect)), np.log(1-h)))

        J.append(cost)

    # Part 2: Implement the backpropagation algorithm to compute the gradients
    # Theta1_grad and Theta2_grad.
    # You should return the partial derivatives of the cost function with respect
    # to Theta1 and Theta2 in Theta1_grad and Theta2_grad, respectively.

        # delta at the output layer
        delta_3 = (h - y_vect)
        # delta for the hidden layer
        # remove delta_2_0 (gradients of bias units) by doing Theta2[:,1:]
        delta_2 = np.dot(np.transpose(Theta2[:,1:]), delta_3) * sigmoidGradient(z_2)
        # Accumulate the gradients (DELTA)
        Theta1_grad = Theta1_grad + \
            np.dot(np.vstack(delta_2), np.transpose(np.vstack(act_1)))

        Theta2_grad = Theta2_grad + \
            np.dot(np.vstack(delta_3), np.transpose(np.vstack(act_2)))

    # Regularized gradient for tall
    capital_delta1 = 1/m * Theta1_grad + np.dot(lambda_param/m, Theta1)
    capital_delta2 = 1/m * Theta2_grad + np.dot(lambda_param/m, Theta2)

    # Adjust for the first column of Theta. Not regularization for j=0
    capital_delta1[:,0] = 1/m * Theta1_grad[:,0]
    capital_delta2[:,0] = 1/m * Theta2_grad[:,0]

    # Regularized term
    # Take out the bias term in the first column
    regul_term = lambda_param/(2*m)*(np.sum(np.power(Theta1[:, 1:], 2)) +
                                     np.sum(np.power(Theta2[:, 1:], 2)))

    J = sum(J) + regul_term

    # Unroll gradients
    grad = []
    grad.extend((list(capital_delta1.flatten()) +
                 list(capital_delta2.flatten())))

    return J, grad


'''
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
'''
