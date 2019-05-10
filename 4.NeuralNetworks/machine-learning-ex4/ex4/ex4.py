# Machine Learning Online Class - Exercise 4 Neural Network Learning
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random
import numpy as np
from displayData import displayData
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients
from scipy import optimize
from predict import predict

# Setup the parameters you will use for this exercise
input_layer_size = 400   # 20x20 Input Images of Digits
hidden_layer_size = 25    # 25 hidden units
num_labels = 10           # 10 labels, from 1 to 10
# (note that we have mapped "0" to label 10)

# =========== Part 1: Loading and Visualizing Data =============

# Load Training Data
print('Loading and Visualizing Data ...\n')


data = loadmat('ex4data1.mat')
X = data['X']
y = data['y']
m = X.shape[0]

# Randomly select 100 data points to display
sel = np.random.permutation(m)
sel = sel[:100]

displayData(X[sel])

# Since python accept zero index, map the digit zero to value 0
#np.place(y, y == 10, 0)

input('Program paused. Press enter to continue.\n')

# ================ Part 2: Loading Parameters ================
# Load some pre-initialized neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2

weights = loadmat('ex4weights.mat')

Theta1 = weights['Theta1']  # 25x401
Theta2 = weights['Theta2']  # 10x26

# Unrolling Parameters to Vector for Implementation
nn_params = []
nn_params.extend((list(Theta1.flatten()) +
                  list(Theta2.flatten())))

# ================ Part 3: Compute Cost (Feedforward) ================
#  1. start by implementing the feedforward part of the
#     neural network that returns the cost only (nnCostFunction.py)
#  2. verify that your implementation is correct by verifying that you
#     get the same cost as us for the fixed debugging parameters.

print('\nFeedforward Using Neural Network ...\n')

# Weight regularization parameter (we set this to 0 here).
lambda_param = 0

J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                         num_labels, X, y, lambda_param)

print('Cost at parameters (loaded from ex4weights): {:.6f} '.format(float(J)))
print('\n(this value should be about 0.287629)\n')

input('\nProgram paused. Press enter to continue.\n')

# =============== Part 4: Implement Regularization ===============

print('\nChecking Cost Function (w/ Regularization) ... \n')

# Weight regularization parameter

lambda_param = 1

J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                         num_labels, X, y, lambda_param)

print('Cost at parameters (loaded from ex4weights): {:.6f} '.format(float(J)))
print('\n(this value should be about 0.383770)\n')

input('Program paused. Press enter to continue.\n')

# ================ Part 5: Sigmoid Gradient  ================
print('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient([-1, -0.5, 0, 0.5, 1])

print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ')
print('{}'.format(g))
print('\n\n')

input('Program paused. Press enter to continue.\n')

# ================ Part 6: Initializing Pameters ================
# Implment a two layer neural network that classifies digits.
# Start by implementing a function to initialize the weights of the neural network

print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = []
initial_nn_params.extend((list(initial_Theta1.flatten()) +
                          list(initial_Theta2.flatten())))

initial_nn_params = np.array(initial_nn_params)

# =============== Part 7: Implement Backpropagation ===============
# add in nnCostFunction.py to return the partial derivatives of the parameters.

print('\nChecking Backpropagation... \n')

# Check gradients by running checkNNGradients
checkNNGradients()

input('\nProgram paused. Press enter to continue.\n')

# =============== Part 8: Implement Regularization ===============

print('\nChecking Backpropagation (w/ Regularization) ... \n')

#  Check gradients by running checkNNGradients
lambda_param = 3
checkNNGradients(lambda_param)

# Also output the costFunction debugging values

debug_J, _ = nnCostFunction(nn_params, input_layer_size,
                            hidden_layer_size, num_labels, X, y, lambda_param)

print('\n\nCost at (fixed) debugging parameters (w/ lambda = {0}): {1} '
      '\n(for lambda = 3, this value should be about 0.576051)\n\n'.format(lambda_param, debug_J))

input('Program paused. Press enter to continue.\n')

# =================== Part 8: Training NN ===================
# Train your neural network using advanced optimizer optimize.minimize from scipy
# Advanced optimizers are able to train the cost functions efficiently as
# long as we provide the gradient computations.

print('\nTraining Neural Network... \n')

#  Try different values of lambda
# set the regularization lambda to a smaller value (closer to overfitting)

lambda_param = 1

myargs = (input_layer_size, hidden_layer_size, num_labels, X, y, lambda_param)

# Try different values of MaxIter to see how more training helps
# set MaxIter to a higher number (closer to overfitting)

result = optimize.minimize(nnCostFunction, initial_nn_params,
                                    args=myargs, method='CG', jac=True,
                                    options={'disp': True, 'maxiter': 50})

nn_params = result['x']
# Obtain Theta1 and Theta2 back from nn_params

Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],
                  (hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[(hidden_layer_size*(input_layer_size+1))::],
                  (num_labels, (hidden_layer_size + 1)))

input('Program paused. Press enter to continue.\n')

## ================= Part 9: Visualize Weights =================
# You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

print('\nVisualizing Neural Network... \n')

displayData(Theta1[:,1:])

input('\nProgram paused. Press enter to continue.\n')

## ================= Part 10: Implement Predict =================
# use the neural network to predict the labels of the training set. 
# This lets you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)

y = np.int_(y)

print('\nTraining Set Accuracy: %1f\n' % (np.mean((pred == y.flatten()) * 100)))
