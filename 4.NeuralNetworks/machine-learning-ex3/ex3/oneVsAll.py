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
    (m, n) = X.shape

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
        c = np.array([1 if elem_y == k else 0 for elem_y in y]).reshape(m, 1)

        # Initialize fitting parameters
        initial_theta = np.zeros((n + 1))

        output = optimize.minimize(lrCostFunction, initial_theta,
                                   args=(X, c, lambda_par), method='CG', jac=True, 
                                   options={'maxiter': 400})

        print('output message:', output.message)
        print('k:', k)

        all_theta[k, :] = output.x

    return all_theta


'''

% Hint: theta(:) will return a column vector.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
'''
