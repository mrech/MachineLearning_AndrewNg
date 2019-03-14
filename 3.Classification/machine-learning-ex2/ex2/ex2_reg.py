# Machine Learning Online Class - Exercise 2: Logistic Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from plotData import *
from mapFeature import *
from costFunctionReg import *
from plotDecisionBoundary import *
from predict import *
import os
os.chdir('/home/morena/MachineLearning/AndrewNg_Python/3.Classification/machine-learning-ex2/ex2')

# Load Data
# The first two columns contains the X values and the third column
# contains the label (y).
data = pd.read_csv('ex2data2.txt', header=None)
X = data.iloc[:, 0:2]
y = data.iloc[:, 2]

plotData(X, y)

# Put some labels
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
# plt.gca() to access the Axes instance
plt.gca().legend(('y = 1 - Accepted ', 'y = 0 - Rejected'))
plt.show()

# =========== Part 1: Regularized Logistic Regression ============

'''
In this part, you are given a dataset with data points that are not
linearly separable. However, you would still like to use logistic
regression to classify the data points.

To do so, you introduce more features to use -- in particular, you add
polynomial features to our data matrix (similar to polynomial
regression).
'''

# Add Polynomial Features
X = mapFeature(X.iloc[:, 0], X.iloc[:, 1])

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1]))

# Set regularization parameter lambda to 1
RegParam = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
[cost, grad] = costFunctionReg(initial_theta, X, y, RegParam)

print('Cost at initial theta (zeros): %f\n' % cost)
print('Expected cost (approx): 0.693\n')

print('Gradient at initial theta (zeros) - first five values only:\n')
for i in range(5):
    print('gradient_%i: %.4f' % (i+1, grad[i]))

print('\nExpected gradients (approx) - first five values only:\n')
print('0.0085\n0.0188\n0.0001\n0.0503\n0.0115\n')

input('\nProgram paused. Press enter to continue.\n')

# Compute and display cost and gradient
# with all-ones theta and lambda (RegParam) = 10
test_theta = np.ones((X.shape[1]))

[cost, grad] = costFunctionReg(test_theta, X, y, 10)

print('\nCost at test theta (with lambda = 10): %f\n' % cost)
print('Expected cost (approx): 3.16\n')

print('Gradient at test theta - first five values only:\n')
for i in range(5):
    print('gradient_%i: %.4f' % (i+1, grad[i]))

print('Expected gradients (approx) - first five values only:\n')
print('0.3460\n0.1614\n0.1948\n0.2269\n0.0922\n')

input('\nProgram paused. Press enter to continue.\n')

# ============= Part 2: Regularization ainitial_theta = np.zeros((X.shape[1]))nd Accuracies =============
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?

#  Try the following values of lambda (0, 1, 10, 100).

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1]))

# Set regularization parameter lambda to 1 (you should vary this)
RegParam = 1

# Optimize
output = minimize(costFunctionReg, initial_theta, args=(
    X, y, RegParam), method='bfgs', jac=True, options={'maxiter': 400})

theta = output.x
J = output.fun
exit_flag = output.message

# Plot Boundary
plotDecisionBoundary(theta, X, y)
plt.title(f'lamda = {RegParam}')
# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
# plt.gca() to access the Axes instance
plt.gca().legend(('y = 1 - Accepted ', 'y = 0 - Rejected', 'Decision boundary'))
plt.show()

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %.1f\n' % (np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): 83.1 (approx)\n')
