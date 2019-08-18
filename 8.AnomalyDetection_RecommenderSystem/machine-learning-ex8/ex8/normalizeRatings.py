# NORMALIZERATINGS Preprocess data by subtracting mean rating for every 
# movie (every row) rated

def normalizeRatings(Y, R):
    '''
    [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
    has a rating of 0 on average, and returns the mean rating in Ymean.
    '''
    import numpy as np

    m, n = np.shape(Y)

    Ymean = np.zeros((m,1))
    Ynorm = np.zeros(np.shape(Y))

    for i in range(m):
        idx = np.where(R[i, :] == 1)
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean