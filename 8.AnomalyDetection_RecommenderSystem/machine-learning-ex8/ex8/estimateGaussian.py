# ESTIMATEGAUSSIAN This function estimates the parameters of a 
# Gaussian distribution using the data in X

def estimateGaussian(X):
    '''
    [mu sigma2] = estimateGaussian(X), 
    The input X is the dataset with each n-dimensional data point in one row
    The output is an n-dimensional vector mu, the mean of the data set
    and the variances sigma^2, an n x 1 vector
    '''
    
    import numpy as np

    # Useful variables
    m, n = X.shape

    mu = np.mean(X, axis = 0)
    sigma2 = np.var(X, axis = 0)

    return mu, sigma2



'''
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
'''