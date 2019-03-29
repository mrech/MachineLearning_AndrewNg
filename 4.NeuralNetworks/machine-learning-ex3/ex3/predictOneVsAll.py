# Predict the label for a trained one-vs-all classifier. The labels 
# are in the range 1..K, where K = all_theta.shape[0]. 
from sigmoid import sigmoid
import numpy as np

def predictOneVsAll(all_theta, X):
    '''
    p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
    for each example in the matrix X. Note that X contains the examples in
    rows. all_theta is a matrix where the i-th row is a trained logistic
    regression theta vector for the i-th class. You should set p to a vector
    of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
    for 4 examples)
    '''

    m = X.shape[0]

    # add ones to the X data matrix
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # Initialize prediction
    p = np.zeros((m, 1))

    all_theta = np.transpose(all_theta)

    p = np.argmax(sigmoid(np.dot(X, all_theta)), axis=1)

    return p