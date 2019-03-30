# Predict the label of an input given a trained neural network


def predict(Theta1, Theta2, X):
    '''
    p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    '''
    import numpy as np
    from sigmoid import sigmoid

    # for X equal vector column insert row dimension
    # when predicting only ONE example
    if X.ndim == 1:
        X = X.reshape(1, X.shape[0])

    # useful values
    m = X.shape[0]

    # initialize p (vector containing labels between 1 to num_labels.)
    # to be sure that you index X correctly you need a column vector
    p = np.zeros((m, 1))

    # add ones to the X data matrix
    X = np.concatenate((np.ones((m, 1)), X), axis=1)  # R:5000x401

    # calculate activation in layer 2
    a2 = sigmoid(np.dot(X, np.transpose(Theta1)))  # R: 5000X25

    # add the bias unit equal to 1
    a2 = np.hstack((np.ones((m, 1)), a2))  # R: 5000x26

    # calculate activation in layer 3, which returns
    # associated predictions for every exaples
    a3 = sigmoid(np.dot(a2, np.transpose(Theta2)))  # R: 5000x10

    # the prediction from the neural network will be
    # the label that has the largest output
    # argmax axis=1 returns the max for each column > R: 5000
    p = np.argmax(a3, axis=1)

    # offsets python's zero notation !!! (print p and y to see the pattern)
    return p + 1


'''
IMPORTANT:
In the last equality, we used the fact that a^T*b = b^T*a if a and b are vectors.
This allows us to compute the products θ^T* x^(i) for all our examples i in one
line of code.

P.S. equally good: a2 = sigmoid(np.dot(Theta1, np.transpose(X))) # R: 25x5000
                    ...
                   p = np.argmax(a3, axis=1)  # max for each row
'''
