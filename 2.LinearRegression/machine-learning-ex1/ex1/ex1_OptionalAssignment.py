# Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from featureNormalize import *
from gradientDescent import *
from computeCost import *
import os
os.getcwd()
os.chdir('/home/morena/MachineLearning/AndrewNg_Python/2.LinearRegression/machine-learning-ex1/ex1')

# ================ Part 1: Feature Normalization ================
print('Loading data ...\n')

# Load Data
data = pd.read_csv('ex1data2.txt', header=None)

X = data[[0, 1]]
y = data[[2]]
m = len(y)

# Print out some data points
print('First 10 examples from the dataset: \n')
print("X ={},\n\ny ={}".format(X.head(11), y[0:11]))


input('\nProgram paused. Press enter to continue.\n')

# Scale features and set them to zero mean

print('\nNormalizing Features ...')
print('House sizes are about 1000 times the number of bedrooms.\n')

[X, mu, sigma] = featureNormalize(X)

# Add a column of ones to X (intercept term)
X = np.hstack((np.ones((m, 1)), X))

# ================ Part 2: Gradient Descent ================

print('Running gradient descent ...\n')

# Choose some alpha value
'''
trying values of the learning rate α on a log-scale, at multiplicative
steps of about 3 times the previous value (i.e., 0.3, 0.1, 0.03, 0.01 and so on).
'''

# Initiate list where to store the cost function for different alpha
J_history_alpha = []

for alpha in [0.3, 0.1, 0.03, 0.01]:

    num_iters = 50
    # Init Theta and Run Gradient Descent
    theta = np.zeros((X.shape[1], 1))
    y = np.array(data[1])
    (theta, J_history) = gradientDescent(X, y, theta, alpha, num_iters)
    J_history_alpha.append(J_history)


# Plot the convergence graph
plt.plot(range(num_iters), J_history_alpha[0]*10**10, 'g', label='alpha = 0.3')
plt.plot(range(num_iters), J_history_alpha[1]*10**10, 'b', label='alpha = 0.1')
plt.plot(range(num_iters),
         J_history_alpha[2]*10**10, 'r', label='alpha = 0.03')
plt.plot(range(num_iters),
         J_history_alpha[3]*10**10, 'k', label='alpha = 0.01')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.title('Convergence of gradient descent with an appropriate learning rate', y=1.05)
plt.legend()
plt.savefig('CovergenceGraph.png')
plt.show()

# Using the learning rate 0.1
alpha = 0.1
num_iters = 50
# Init Theta and Run Gradient Descent
theta = np.zeros((X.shape[1], 1))
y = np.array(data[1])
(theta, J_history) = gradientDescent(X, y, theta, alpha, num_iters)

# Display gradient descent's result
print('Theta computed from gradient descent: \n')
print(f'{theta}\n')
print('\n')

# Estimate the price of a 1650 sq-ft, 3 br house
# Hint: At prediction, make sure you do the same feature normalization.

X_Test = np.array([1650, 3])

# Normalize
X_Test = (X_Test-mu)/sigma
X_Test = np.array(X_Test).reshape(1, len(X_Test))

# Add first column of all-ones
X_Test = np.hstack((np.ones((X_Test.shape[0], 1)), X_Test))

# Prediction
price = X_Test @ theta
print(f"""Predicted price of a 1650 sq-ft, 3 br house
(using gradient descent):\n{price}\n""")

print('Program paused. Press enter to continue.\n')


'''
%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = 0; % You should change this


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

'''
