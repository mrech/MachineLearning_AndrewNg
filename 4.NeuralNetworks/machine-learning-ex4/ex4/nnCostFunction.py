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
        act_2 = np.append(1, act_2)
        z_3 = np.dot(Theta2, act_2)
        h = sigmoid(z_3)

        y_vect = np.zeros(num_labels)
        y_vect[y[i]-1] = 1

        cost = -1/m * (np.dot(np.transpose(np.vstack(y_vect)), np.log(h)) +
                       np.dot(np.transpose(np.vstack(1-y_vect)), np.log(1-h)))

        J.append(cost)

    # Unroll gradients
    grad = []
    grad.extend((list(Theta1_grad.flatten()) +
                 list(Theta2_grad.flatten())))

    return sum(J), grad


'''

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
'''
