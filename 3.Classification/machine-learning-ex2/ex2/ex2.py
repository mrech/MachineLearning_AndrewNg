# Machine Learning Online Class - Exercise 2: Logistic Regression

from scipy.optimize import minimize
from costFunction import *
from plotData import *
from plotDecisionBoundary import *
from predict import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load Data
data = pd.read_csv('ex2data1.txt', header=None)
# The first two columns contains the exam scores
# the third column contains the label.

X = data.iloc[:, 0:2]
y = data.iloc[:, 2]

# ==================== Part 1: Plotting ====================
# We start the exercise by first plotting the data to understand the
# the problem we are working with.

print('Plotting data with + indicating (y = 1) examples')
print('and o indicating (y = 0) examples.\n')

plotData(X, y)
plt.show()

input('\nProgram paused. Press enter to continue.\n')

# ============ Part 2: Compute Cost and Gradient ============
# In this part of the exercise, you will implement the cost and gradient
# for logistic regression.

# Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = np.shape(X)

# Add intercept term to x and X_test
X[-1] = 1
# Re-ordering columns
X = X.reindex(sorted(X.columns), axis=1)

# Initialize fitting parameters
initial_theta = np.zeros((n + 1))

# Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y)

print(f'Cost at initial theta (zeros): {float(cost)}')
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros):')
for i in range(n+1):
    print(" grad%i = %f" % (i, grad[i]))

print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])

[cost, grad] = costFunction(test_theta, X, y)

print(f'\nCost at test theta: {float(cost)}\n')
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta:')
for i in range(n+1):
    print(" grad%i = %f" % (i, grad[i]))

print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

input('\nProgram paused. Press enter to continue.\n')

# ============= Part 3: Optimizing using minimize  =============
#  In this exercise, you will use a built-in function (minimize) to find the
#  optimal parameters theta.

output = minimize(costFunction, initial_theta,
                  args=(X, y), method='bfgs', jac=True, options={'maxiter': 400})

# Print theta to scree
print(f'Cost at theta found by minimize: {output.fun}\n')
print('Expected cost (approx): 0.203\n')
print('theta: \n')
for i in range(n+1):
    print(" theta%i = %f" % (i, output.x[i]))

print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

theta = output.x

# Plot Boundary
plotDecisionBoundary(theta, X, y)
plt.show()

input('\nProgram paused. Press enter to continue.\n')

# ============== Part 4: Predict and Accuracies ==============
'''
After learning the parameters, you'll like to use it to predict the outcomes
on unseen data. In this part, you will use the logistic regression model
to predict the probability that a student with score 45 on exam 1 and 
score 85 on exam 2 will be admitted.
'''

prob = sigmoid(np.dot(np.array([1, 45, 85]), theta.reshape((n+1, 1))))

print('For a student with scores 45 and 85,')
print(f'we predict an admission probability of {float(prob)}\n')
print('Expected value: 0.775 +/- 0.002\n\n')

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %.2f\n' % (np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.0\n')
print('\n')
