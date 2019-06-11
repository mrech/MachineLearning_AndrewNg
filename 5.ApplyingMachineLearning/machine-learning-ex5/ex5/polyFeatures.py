# POLYFEATURES Maps X (1D vector) into the p-th power


def polyFeatures(X, p):
    '''
    [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
    maps each example into its polynomial features where
    X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p]
    '''
    import numpy as np
    # You need to return the following variables correctly.
    X_poly = np.zeros((X.size, p))

    for i in range(X.size):
        for j in range(p):
            X_poly[i, j] = X[i]**(j+1)

    return X_poly
