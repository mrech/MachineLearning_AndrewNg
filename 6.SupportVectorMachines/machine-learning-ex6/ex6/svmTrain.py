
def svmTrain(X, y, C, kernelFunction, tol=1e-3, max_passes=-1, sigma=0.1):
    ''' Train SVM classifier '''

    from sklearn import svm
    import numpy as np
    from gaussianKernelGramMatrix import gaussianKernelGramMatrix

    y = y.flatten()

    if kernelFunction == 'gaussian':
        # https://scikit-learn.org/stable/modules/svm.html#custom-kernels
        # Define linear kernels by precomputing the Gram matrix
        clf = svm.SVC(C=C, kernel='precomputed', tol=tol, max_iter=max_passes)
        return clf.fit(gaussianKernelGramMatrix(X,X, sigma=sigma), y)
    
    # alternative: Define linear kernels by giving the kernel a python function
    # issue: it does not provide 'coef_' useful to 'visualizeBoundaryLinear'

    #elif kernelFunction == 'linear':
    #    clf = svm.SVC(C = C, kernel=lk.linearKernel, tol=tol, max_iter=max_passes)
    #    return clf.fit(X, y)

    else: # this works with linear and rbf. It doesn't use custom implementation of linear kernel and gaussian kernel
        clf = svm.SVC(C = C, kernel=kernelFunction, tol=tol, max_iter=max_passes)
        return clf.fit(X, y)