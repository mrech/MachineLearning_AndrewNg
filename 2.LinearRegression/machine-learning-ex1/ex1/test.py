# Random Test Cases
import numpy as np
from gradientDescent import *
from computeCost import *
from warmUpExercise import *
from featureNormalize import *
from normalEqn import *
import os
os.chdir('/home/morena/MachineLearning/AndrewNg_Python/2.LinearRegression/machine-learning-ex1/ex1')

# Inputs to the functions
X1 = np.transpose(
    np.array([np.ones(20), np.exp(1) + np.exp(2) * np.arange(0.1, 2.1, 0.1)]))
Y1 = np.transpose(np.array([X1[:, 1]+np.sin(X1[:, 0])+np.cos(X1[:, 1])]))
X2 = np.transpose(np.array([X1[:, 1]**0.5, X1[:, 1]**0.25]))
X2 = np.concatenate((X1, X2), axis=1)
Y2 = np.array(Y1**0.5 + Y1)

# WarmUpExercise
print('Print WarmUpExercise:\n{}'.format(warmUpExercise(5)))

# computeCost with one variable
print('Print computeCost with one variable:\n{}'.format(computeCost(X1, Y1.transpose(),
                                                                    np.array([0.5, -0.5]).transpose())))

# gradientDescent with one variable
(theta, J_history) = gradientDescent(
    X1, Y1[:, 0], np.array([0.5, -0.5]).transpose(), 0.01, 10)
print('theta_single = {}, J_history_single = {}'.format(theta, J_history))

# Feature Normalization
[X_norm, mu, sigma] = featureNormalize(X2[:, 1:3])
print('X_norm:\n{}\nmu:{}\nsigma{}'.format(X_norm, mu, sigma))

# computeCost with multiple variables
costMulti = computeCost(X2, Y2.transpose(), np.array(
    [0.1, 0.2, 0.3, 0.4]).transpose())
print('Print computeCost with multiple variables\n{}'.format(costMulti))

# gradientDescent with multiple variables
(theta_multi, J_history_multi) = gradientDescent(
    X2, Y2[:, 0], np.array([-0.1, -0.2, -0.3, -0.4]).transpose(), 0.01, 10)
print('theta_multi = {}, J_history_multi = {}'.format(
    theta_multi, J_history_multi))

# Normal Equation
print('Print theta from normal equation:\n{}'.format(normalEqn(X2, Y2[:, 0])))
