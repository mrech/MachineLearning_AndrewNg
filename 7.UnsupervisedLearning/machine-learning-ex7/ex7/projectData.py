
# PROJECTDATA Computes the reduced data representation when projecting only 
# on to the top k eigenvectors

def projectData(X, U, K):
    '''
    Z = projectData(X, U, K) computes the projection of 
    the normalized inputs X into the reduced dimensional space spanned by
    the first K columns of U. It returns the projected examples in Z.
    '''

    import numpy as np

    # You need to return the following variables correctly.
    Z = np.zeros((np.size(X, 0), K))

    Z = np.dot(X, U[:, :K])
    
    return Z