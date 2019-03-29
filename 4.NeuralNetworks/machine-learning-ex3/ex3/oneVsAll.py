# trains multiple logistic regression classifiers and returns all
# the classifiers parameters in a matrix all_theta, where the i-th
# row of all_theta corresponds to the learned logistic regression
# parameters for label i

from lrCostFunction import lrCostFunction
import numpy as np
from scipy import optimize


def oneVsAll(X, y, num_labels, lambda_par):
    '''
    [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
    logistic regression classifiers and returns each of these classifiers
    in a matrix all_theta, where the i-th row of all_theta corresponds 
    to the classifier for label i
    '''

    # X dimensions
    m, n = X.shape

    # initiate all_theta
    all_theta = np.zeros((num_labels, n+1))

    # add ones to the X data matrix
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # train num_labels logistic regression classifiers
    # using advance optimization.

    for k in range(num_labels):

        # train the classifier for class k ∈ {1, ..., K}
        # new vector with 1 for the specific K, 0 for all the remaining
        # using list comprehension

        print("Training {:d} out of {:d} categories...".format(
            k+1, num_labels))

        # True stands for function with gradient/jac paremeter
        myargs = (X, (y == k).astype(int), lambda_par)

        # Initialize fitting parameters
        initial_theta = np.zeros((n + 1, 1))

        theta = optimize.minimize(lrCostFunction, initial_theta,
                                  args=myargs, method='CG', jac=True,
                                  options={'disp': True, 'maxiter': 50})

        all_theta[k, :] = theta['x']

    return all_theta
