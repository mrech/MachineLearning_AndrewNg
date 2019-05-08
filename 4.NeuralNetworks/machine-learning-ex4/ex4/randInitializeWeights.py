# RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
# incoming connections and L_out outgoing connections


def randInitializeWeights(L_in, L_out):
    '''
    W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
    of a layer with L_in incoming connections and L_out outgoing 
    connections. 

    Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    the first column of W handles the "bias" terms
    '''

    import numpy as np

    W = np.zeros((L_out, 1 + L_in))

    epsilon_init = np.sqrt(6)/np.sqrt(L_in + L_out)

    W = np.random.uniform(0, 1, size=((L_out, 1 + L_in))) *\
        (2*epsilon_init) - epsilon_init

    return W
