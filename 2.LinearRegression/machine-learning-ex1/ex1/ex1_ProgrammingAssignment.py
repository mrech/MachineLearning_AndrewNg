# Machine Learning Online Class - Exercise 1: Linear Regression
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gradientDescent import *
from computeCost import *
from plotData import *
from warmUpExercise import *

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

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1)...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i][j] = computeCost(X, y, t)

# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.transpose()

fig = plt.figure(figsize=plt.figaspect(0.3))

# ==============
# Surface Plot
# ==============

ax = fig.add_subplot(1, 2, 1, projection='3d')

# Make data
# return coordinate matrices from coordinate vectors
theta0, theta1 = np.meshgrid(theta0_vals, theta1_vals)

# Plot the surface
surf = ax.plot_surface(theta0, theta1, J_vals, cmap=plt.cm.jet)

# customize the z axis
ax.set_zlim(0, 800)

# labels and title
plt.xlabel(r"$\theta_0$")
plt.xticks(np.arange(min(theta0_vals), max(theta0_vals+1), 5.0))
plt.ylabel(r"$\theta_1$")
plt.title('(a) Surface')

# ==============
# Contour plot
# ==============

# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = fig.add_subplot(1, 2, 2)
cset = ax.contour(theta0_vals, theta1_vals, J_vals,
                  np.logspace(-2, 3, 20))  # 10**-2, 10**3
ax.clabel(cset, fontsize=9, inline=1)

# labels and title
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.title('(b) Contour, showing minimum')

# plot the minimum
plt.plot(theta[0], theta[1], 'rx', linewidth=2, markersize=10)


plt.savefig('Cost_function.png')
plt.show()
