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

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)

X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
X_poly = np.hstack((np.ones((X_poly.shape[0],1)), X_poly)) # Add Ones

# Map X_poly_test and normalize (using mu and sigma from the training set)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test/sigma
X_poly_test = np.hstack((np.ones((X_poly_test.shape[0],1)), X_poly_test)) # Add Ones

# Map X_poly_val and normalize (using mu and sigma from training set)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val/sigma
X_poly_val = np.hstack((np.ones((X_poly_val.shape[0],1)), X_poly_val)) # Add Ones

print('Normalized Training Example 1:\n')
print('  {}  \n'.format(list(X_poly[1, :])))

input('\nProgram paused. Press enter to continue.\n')

'''
%% =========== Part 7: Learning Curve for Polynomial Regression =============
%  Now, you will get to experiment with polynomial regression with multiple
%  values of lambda. The code below runs polynomial regression with
%  lambda = 0. You should try running the code with different values of
%  lambda to see how the fit and learning curve change.
%

lambda = 0;
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

figure(2);
[error_train, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 8: Validation for Selecting Lambda =============
%  You will now implement validationCurve to test various values of
%  lambda on a validation set. You will then use this to select the
%  "best" lambda value.
%

[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;
'''
