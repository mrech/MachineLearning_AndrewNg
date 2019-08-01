# DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
# where you select the optimal (C, sigma) learning parameters to use for SVM
# with RBF kernel

def dataset3Params(X, y, Xval, yval):
    '''
    C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
    sigma. You should complete this function to return the optimal C and 
    sigma based on a cross-validation set.
    '''

    from svmTrain import svmTrain
    import numpy as np
    import gaussianKernelGramMatrix as gkgm

    # initialize prediction error which collect the errors for each of the
    # 64 models together with its sigma and C hyper-parameters
    predictionErrors = np.zeros((64, 3))
    Counter = 0

    for sigma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:

            # Train the model on training set
            model = svmTrain(X, y, C=C, kernelFunction="gaussian", sigma=sigma)
            # Evaluate model on the cross-validation set
            Z = model.predict(gkgm.gaussianKernelGramMatrix(Xval, X))
            # Compute the prediction errors
            predictionErrors[Counter, 0] = np.mean(
                (Z != yval.flatten()).astype(int))
            # store corresponding sigma and C
            predictionErrors[Counter, 1] = sigma
            predictionErrors[Counter, 2] = C

            # Move counter up
            Counter += 1

    # extract the row number with the lower error
    # since its a tuple, we pick the first min found inside the tuple
    minIndex = np.where(predictionErrors[:, 0] == np.min(predictionErrors[:, 0]))[0][0]

    sigma = predictionErrors[minIndex, 1]
    C = predictionErrors[minIndex, 2]

    return sigma, C


'''
ALTERNATIVE USING 'GridSearchCV' from 'klearn.model_selection'
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from scipy.io import loadmat
from sklearn.svm import SVC

# Load from ex6data3
data = loadmat('ex6data3.mat')
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1.00000000e+02, 3.33333333e+01, 1.00000000e+01, 3.33333333e+00,
       1.00000000e+00, 3.33333333e-01, 1.00000000e-01, 3.33333333e-02],
       'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]}]


scores = ['precision', 'recall']

for score in scores:
    print('# Tuning hyper-parameters for %s' % score)
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
    clf.fit(X, y.flatten())
    print('\nBest parameters set found on development set:')
    print(clf.best_params_)
    print('\nGrid scores on development set:')
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()    
    y_true, y_pred = yval, clf.predict(Xval)
    print(classification_report(y_true, y_pred))
    print()

# warnings
set(list(yval)) - set(list(y_pred))
'''
