
# RECOVERDATA Recovers an approximation of the original data when using the 
# projected data

def recoverData(Z, U, K):
    '''
    X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
    original data that has been reduced to K dimensions. It returns the
    approximate reconstruction in X_rec.
    '''

    import numpy as np

    # You need to return the following variables correctly.
    X_rec = np.zeros((np.size(Z, 0), np.size(U, 0)))

    X_rec = np.dot(Z, np.transpose(U[:,:K]))

    return X_rec