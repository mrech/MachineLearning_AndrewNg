# EMAILFEATURES takes in a word_indices vector and produces a feature vector
# from the word indices

def emailFeatures(word_indices):
    '''
    x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
    construct a binary feature vector that indicates whether a particular
    word occurs in the email.

    The feature vector looks like:
    x = [ 0 0 0 0 1 0 0 0 ... 0 0 0 0 1 ... 0 0 0 1 0 ..];
    '''

    import numpy as np

    # Total number of words in the dictionary
    n = 1899

    # You need to return the following variables correctly.
    x = np.zeros((n, 1))

    for num in word_indices:
        x[num - 1] = 1 # adj for python indexing starting at 0

    return x