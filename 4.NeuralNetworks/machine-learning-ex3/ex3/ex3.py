# Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from lrCostFunction import *
from oneVsAll import oneVsAll, lrCostFunction, optimize
from predictOneVsAll import predictOneVsAll, sigmoid

# Setup the parameters you will use for this part of the exercise

input_layer_size = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10


# =========== Part 1: Loading and Visualizing Data =============
# We start the exercise by first loading and visualizing the dataset.
# You will be working with a dataset that contains handwritten digits.

# Load Training Data
data = loadmat('ex3data1.mat')  # training data stored in arrays X, y
X = data['X']  # X_5000x400
y = data['y']  # y_5000x1

# crucial step in getting good performance!
# changes the dimension from (m,1) to (m,)
# otherwise the minimization isn't very effective...
y=y.flatten() 

# (note that we have mapped "0" to label 10)
np.place(y, y == 10, 0)

'''
5000 training examples, each is a 20x20 pixel grayscale image of the digit.
y contains labels for the training set.
'''

# visualizing the data
# create a figure and a set of subplots (in specific 10 by 10)
figure, axes = plt.subplots(10, 10)

# for each subplots in the row
for i in range(10):
    for j in range(10):
        axes[i, j].imshow(X[np.random.randint(X.shape[0])].
                          reshape((20, 20), order='F'), cmap='gray')
        axes[i, j].axis('off')

plt.show()

input('Program paused. Press enter to continue.\n')

# ============ Part 2a: Vectorize Logistic Regression ============
# Test case for lrCostFunction

print('\nTesting lrCostFunction() with regularization')

theta_t = np.array([-2, -1, 1, 2])
X_t = np.concatenate((np.ones(5).reshape(5, 1),
                      np.arange(0.1, 1.6, 0.1).reshape((5, 3), order='F')),
                     axis=1)
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3

[J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('\nCost: %f\n' % J)
print('Expected cost: 2.534819\n')

print('Gradients:\n')
for i in range(np.size(grad)):
    print(" grad%i = %f" % (i, grad[i]))

print('\nExpected gradients:\n')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

input('Program paused. Press enter to continue.\n')

# ============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...\n')

lambda_par = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_par)

print('Program paused. Press enter to continue.\n')

# ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X)
print('Training Set Accuracy: %.1f' % (np.mean(pred == y)*100))
