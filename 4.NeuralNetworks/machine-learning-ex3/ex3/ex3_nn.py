# Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from predict import predict
from displayData import dispalyData

# Setup the parameters you will use for this exercise
input_layer_size = 400   # 20x20 Input Images of Digits
hidden_layer_size = 25    # 25 hidden units
num_labels = 10           # 10 labels, from 1 to 10

# =========== Part 1: Loading and Visualizing Data =============
# You will be working with a dataset that contains handwritten digits.

# Load Training Data
print('Loading and Visualizing Data ...\n')

# Load Training Data
data = loadmat('ex3data1.mat')  # training data stored in arrays X, y
X = data['X']  # X_5000x400
y = data['y']  # y_5000x1

# (note that we have mapped "0" to label 10)
y = y.flatten()

# create a figure and a set of subplots (10 by 10)
figure, axes = plt.subplots(10, 10)

# for each subplots in the row
for i in range(10):
    for j in range(10):
        # randomly select a row from X, which represent 20x20pixel image
        axes[i, j].imshow(X[np.random.randint(X.shape[0])].
                          reshape((20, 20), order='F'), cmap='gray')
        axes[i, j].axis('off')

plt.show()

input('Program paused. Press enter to continue.\n')

# ================ Part 2: Loading Pameters ================
# we load some pre-initialized neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2

parameters = loadmat('ex3weights.mat')
Theta1 = parameters['Theta1']  # size 25x401
Theta2 = parameters['Theta2']  # size 10x26

# ================= Part 3: Implement Predict =================
'''
After training the neural network, we would like to use it to predict
the labels. You will now implement the "predict" function to use the
neural network to predict the labels of the training set. This lets
you compute the training set accuracy.
'''

pred = predict(Theta1, Theta2, X)

print('Training Set Accuracy: %.1f' % (np.mean(pred == y)*100))

input('Program paused. Press enter to continue.\n')

# To give you an idea of the network's output, you can also run
# through the examples one at the a time to see what it is predicting.

# Randomly permute examples
m = X.shape[0]
rp = np.random.permutation(m)

for i in range(m):
    # Display
    print('\nDisplaying Example Image\n')
    dispalyData(X[rp[i], :])

    pred = predict(Theta1, Theta2, X[rp[i], :])
    print('\nNeural Network Prediction: %d (digit %d)\n' % (pred, pred % 10))

    # Pause with quit option
    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
        break
