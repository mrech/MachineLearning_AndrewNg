# SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
# outliers

def selectThreshold(yval, pval):
    '''
    [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
    threshold to use for selecting outliers based on the results from a
    validation set (pval) and the ground truth (yval).
    '''

    import numpy as np

    bestEpsilon = 0
    bestF1 = 0

    stepsize = (max(pval) - min(pval)) / 1000

    # Instructions: Compute the F1 score of choosing epsilon as the
    #               threshold and place the value in F1. The code at the
    #               end of the loop will compare the F1 score for this
    #                choice of epsilon and set it to be the best epsilon if
    #               it is better than the current choice of epsilon.

    for epsilon in np.arange(min(pval), max(pval), stepsize):
        
        # predict the anomaly
        prediction = (pval < epsilon) 

        # calculate the F1 score
        tp = sum((prediction == 1) & (yval.flatten() == 1).tolist())
        fp = sum((prediction == 1) & (yval.flatten() == 0).tolist())
        fn = sum((prediction == 0) & (yval.flatten() == 1).tolist())

        # RuntimeWarning handling due to 0/0
        # CASE: when the algorithm classify everyhting as NO ANOMALY
         
        if tp == 0 & fp == 0:
                F1 = 0 
        else:        
                prec = tp/(tp+fp)
                rec = tp/(tp+fn)

                F1 = (2*prec*rec)/(prec+rec)              

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
            
    return bestEpsilon, bestF1