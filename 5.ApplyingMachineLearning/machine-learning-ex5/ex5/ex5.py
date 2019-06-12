# Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from plotFit import plotFit
from validationCurve import validationCurve
import random

# =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment and plot
#  the data.

# Load Training Data
print('Loading and Visualizing Data ...\n')

# Load from ex5data1:
# You will have X, y, Xval, yval, Xtest, ytest in your environment

data = loadmat('ex5data1.mat')
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']

# m = Number of examples
m = X.shape[0]

# Plot training data
plt.plot(X, y, 'rx', linewidth=1.5, markersize=10)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

input('Program paused. Press enter to continue.\n')

# =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear
#  regression.

theta = np.array([[1], [1]])

J = linearRegCostFunction(np.concatenate(
    (np.ones(m).reshape(m, 1), X), axis=1), y, theta, 1)

print('Cost at theta = [1 ; 1]: {:.6f} '
      '\n(this value should be about 303.993192)\n'.format(float(J[0])))

input('Program paused. Press enter to continue.\n')

# =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear
#  regression.

theta = np.array([[1], [1]])

[J, grad] = linearRegCostFunction(np.concatenate(
    (np.ones(m).reshape(m, 1), X), axis=1), y, theta, 1)

print('Gradient at theta = [1 ; 1]:  [{:.6f} ; {:.6f}] '
      '\n(this value should be about [-15.303016; 598.250744])\n'
      .format(float(grad[0]), float(grad[1])))

input('Program paused. Press enter to continue.\n')

# =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train
#  regularized linear regression.

#  Write Up Note: The data is non-linear, so this will not give a great
#                 fit.

#  Train linear regression with lambda = 0
lambda_par = 0

theta = trainLinearReg(np.concatenate(
    (np.ones(m).reshape(m, 1), X), axis=1), y, lambda_par)

print('Visualizing Data and Trained Linear Regression ...\n')

#  Plot fit over the data
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.plot(X, np.concatenate((np.ones(m).reshape(m, 1), X), axis=1)
         @theta, '--', linewidth=2)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

input('Program paused. Press enter to continue.\n')

# =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function.

#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf

lambda_par = 0

error_train, error_val = \
    learningCurve(np.concatenate((np.ones(m).reshape(m, 1), X), axis=1), y,
                  np.concatenate(
                      (np.ones(Xval.shape[0]).reshape(Xval.shape[0], 1), Xval), axis=1), yval,
                  lambda_par)

plt.plot(range(1, m+1), error_train, range(1, m+1), error_val)
plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])
plt.show()

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t{}\t\t{:.6f}\t{:.6f}\n'.format(
        i, float(error_train[i]), float(error_val[i])))

input('Program paused. Press enter to continue.\n')

# =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)

X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))  # Add Ones

# Map X_poly_test and normalize (using mu and sigma from the training set)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test/sigma
X_poly_test = np.hstack(
    (np.ones((X_poly_test.shape[0], 1)), X_poly_test))  # Add Ones

# Map X_poly_val and normalize (using mu and sigma from training set)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val/sigma
X_poly_val = np.hstack(
    (np.ones((X_poly_val.shape[0], 1)), X_poly_val))  # Add Ones

print('Normalized Training Example 1:\n')
print('  {}  \n'.format(list(X_poly[0, :])))

input('\nProgram paused. Press enter to continue.\n')

# =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.

lambda_par = 1
theta = trainLinearReg(X_poly, y, lambda_par)

# Plot training data and fit
plt.figure(1)
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plotFit(min(X), max(X), mu, sigma, theta, p)
plt.title('Polynomial Regression Fit (lambda = {:.6f})'.format(
    float(lambda_par)))
plt.show()

plt.figure(2)
error_train, error_val = \
    learningCurve(X_poly, y, X_poly_val, yval, lambda_par)
plt.plot(range(1, m+1), error_train, range(1, m+1), error_val)
plt.title('Polynomial Regression Learning Curve (lambda = {:.6f})'.format(
    float(lambda_par)))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 100])
plt.legend(['Train', 'Cross Validation'])
plt.show()

print('Polynomial Regression (lambda = {})\n\n'.format(lambda_par))
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t{}\t\t{:.6f}\t{:.6f}\n'.format(
        i, float(error_train[i]), float(error_val[i])))

input('Program paused. Press enter to continue.\n')

# =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.

lambda_vec, error_train, error_val = \
    validationCurve(X_poly, y, X_poly_val, yval)

plt.plot(lambda_vec, error_train, lambda_vec, error_val)
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()

print('lambda\t\tTrain Error\tValidation Error\n')
for i in range(len(lambda_vec)):
    print(' {:.6f}\t{:.6f}\t{:.6f}\n'.format(
        float(lambda_vec[i]), float(error_train[i]), float(error_val[i])))

input('Program paused. Press enter to continue.\n')

# ============== Optinal: Compute test set error ===================
# To get a better indication of the model's performance in the real
# world, it is important to evaluate the 'final' model on test set.
# Compute the test error using the best lambda found on CV.

theta = trainLinearReg(X_poly, y, 3)
error_test, _ = linearRegCostFunction(X_poly_test, ytest, theta, 0)

print('Test Error with lambda = 3: {:.6f}'.format(float(error_test)))

# ==== Optional: learning curves with randomly selected examples ====
# Especially for small training sets, when you plot learning curves
# to debug your algorithms, it is often helpful to average across multiple sets
# of randomly selected examples to determine the training error and cross
# validation error.

# Initialize the error values
error_train_all = np.zeros((m, 50))
error_val_all = np.zeros((m, 50))

for i in range(50):
    # randomly select i examples from the training set and cross validation set
    rand_index = random.sample(list(range(len(X))), len(X))
    X_poly_rand = X_poly[rand_index]
    y_rand = y[rand_index]

    val_rand_index = random.sample(list(range(len(Xval))), len(Xval))
    X_poly_val_rand = X_poly_val[val_rand_index]
    yval_rand = yval[val_rand_index]

    # Learn the parameters theta
    lambda_par = 0.01
    theta = trainLinearReg(X_poly_rand, y_rand, lambda_par)

    # Evaluate the parameters theta on the randomly chosen training and cross validation set
    error_train, error_val = learningCurve(
        X_poly_rand, y_rand, X_poly_val_rand, yval_rand, lambda_par)

    # Store the values
    error_train_all[:,i] = error_train.flatten()
    error_val_all[:,i] = error_val.flatten()

# calculate the Error everage within the trials
error_train_avg = np.mean(error_train_all, axis=1)
error_val_avg = np.mean(error_val_all, axis=1)

# Make a plot
plt.plot(range(1, m+1), error_train_avg, range(1, m+1), error_val_avg)
plt.title('Polynomial Regression Learning Curve (lambda = {:.6f})'.format(
    float(lambda_par)))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 100])
plt.legend(['Train', 'Cross Validation'])
plt.show()
