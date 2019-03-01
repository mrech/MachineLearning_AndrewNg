# Machine Learning Online Class - Exercise 1: Linear Regression
from warmUpExercise import *
from plotData import *
from computeCost import *
from gradientDescent import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''

    gradientDescentMulti.m
    computeCostMulti.m
    featureNormalize.m
    normalEqn.m

'''

# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
print('\nRunning warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')

warmUpExercise(5)

input('\nProgram paused. Press Enter to continue.\n')

# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = pd.read_csv('ex1data1.txt', header=None)

X = np.array(data[0])
y = np.array(data[1])

m = len(y)  # number of training examples

# Plot Data
plotData(X, y)

input('Program paused. Press enter to continue.\n')

# =================== Part 3: Cost and Gradient descent ===================
# Add a column of ones to X (intercept term)
X = np.stack((np.ones(m).reshape(m), X), axis=-1)

# theta in linear regression with one variable represents intercept and slope
# Need one theta for all features in training dataset
(m, n) = X.shape
theta = np.zeros(n).reshape(n, 1)  # initialize fitting parameters to zero

# gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')

# compute and display initial cost
J = computeCost(X, y, theta)
print("With theta = {}\nCost computed = {}\n".format(
    theta.transpose(), round(J, 2)))

# further testing of the cost function
J = computeCost(X, y, [[-1], [2]])
print('\nWith theta = [-1 ; 2]\nCost computed = {}\n'.format(round(J, 2)))
print('Program paused. Press enter to continue. \n')

# Calculate gradientDescent
print('\nRunning Gradient Descent ...\n')
(theta, J_history) = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent:\n')
print('{}\n'.format(theta))

# check the cost function trend over iterations (should never increase)
iters = np.arange(1, iterations+1).reshape(iterations, 1)
plt.plot(J_history, iters)
plt.xlabel('N. Iterations')
plt.ylabel('J(θ) - Cost function')
plt.title('Cost function vs Iteration')
plt.show()

# Plot the linear fit
plt.plot(X, y, 'rx', markersize=10, label='Training data')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:, 1], X@theta, '-', label='Linear regression')
plt.title('Training data with linear regression fit')
plt.legend()
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] @ theta
print('For population = 35,000, we predict a profit of {}\n'.format(predict1*10000))

predict2 = [1, 7] @ theta
print('For population = 70,000, we predict a profit of {}\n'.format(predict2*10000))

print('Program paused. Press enter to continue.\n')

'''
%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
'''
