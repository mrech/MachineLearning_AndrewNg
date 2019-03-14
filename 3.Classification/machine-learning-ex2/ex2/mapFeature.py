# Feature mapping function to polynomial features
import numpy as np
import pandas as pd


def mapFeature(X1, X2):
    '''
    MAPFEATURE(X1, X2) maps the two input features
    to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of 
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Inputs X1, X2 must be the same size
    '''

    # polynomial degree
    degree = 6

    # add the intercept term using DataFrame
    try:
        out = pd.DataFrame(1, index=np.arange(len(X1)), columns=['0'])
    except TypeError:
        out = pd.DataFrame(1, index=np.arange(1), columns=['0'])


    for i in range(1, degree+1):
        for j in range(i+1):
            # add a new column at every interaction
            out[str(i)+'_'+str(j)] = (X1**(i-j))*(X2**j)

    return out
