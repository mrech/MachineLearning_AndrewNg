# Machine Learning Online Class
#  Exercise 6 | Support Vector Machines

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from plotData import *
from svmTrain import *
from sklearn import svm
from visualizeBoundaryLinear import *
from gaussianKernel import *
from visualizeBoundary import *
import linearKernel as lk
from dataset3Params import *

# =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset.

print('Loading and Visualizing Data ...\n')

# Load from ex6data1
data = loadmat('ex6data1.mat')
X = data['X']
y = data['y']

# Plot training data
plotData(X, y)
plt.show()

input('Program paused. Press enter to continue.\n')

# ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.

print('\nTraining Linear SVM ...\n')

# You should try to change the C (PenaltyParameter) value below and see how the decision
# boundary varies (e.g., try C = 1 and C = 100)

C = 1

# model using the linear kernel implementation 
model = svmTrain(X, y, C, kernelFunction = 'linear')

# alternative using the svm implementation
# model = svm.LinearSVC(tol=1e-3, C= C, random_state=0, max_iter=100000)
# model.fit(X,y.ravel())

visualizeBoundaryLinear(X, y, model)
plt.title('SVM Decision Boundary with C = 1')
plt.show()

C = 100

# model using the linear kernel implementation 
model = svmTrain(X, y, C, kernelFunction = 'linear')

# alternative using the svm implementation
# model = svm.LinearSVC(tol=1e-3, C= C, random_state=0, max_iter=100000)
# model.fit(X,y.ravel())

visualizeBoundaryLinear(X, y, model)
plt.title('SVM Decision Boundary with C = 100')
plt.show()

input('Program paused. Press enter to continue.\n')

## =============== Part 3: Implementing Gaussian Kernel ===============
#  Implement the Gaussian kernel to use with the SVM.

print('\nEvaluating the Gaussian Kernel ...\n')
tol=1e-3
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

sim = gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {:.6f} :' \
         '\n\t{:.6f}\n(for sigma = 2, this value should be about 0.324652)\n'.format(float(sigma), float(sim)))

input('Program paused. Press enter to continue.\n')

## =============== Part 4: Visualizing Dataset 2 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 

print('Loading and Visualizing Data ...\n')

data = loadmat('ex6data2.mat')
X = data['X']
y = data['y']

plotData(X, y)
plt.show()

input('Program paused. Press enter to continue.\n')

## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the 
#  SVM classifier.
# 
print('Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...')

# SVM Parameters
# C = 1
# sigma = 0.1

model = svmTrain(X, y, C, kernelFunction = "gaussian")

# alternative without the Gaussian kernel implementation
# Gamma explained behaviour : gamma ~ 1/sigma
#model = svm.SVC(C = 1, kernel = 'rbf', gamma=100)
#model.fit(X, y.ravel())

visualizeBoundary(X, y, model)
plt.show()

input('Program paused. Press enter to continue.\n')

## =============== Part 6: Visualizing Dataset 3 ================
# The following code will load the next dataset into your environment and 
# plot the data. 

print('Loading and Visualizing Data ...\n')

# Load from ex6data3
data = loadmat('ex6data3.mat')
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']

# Plot training data
plotData(X, y)
plt.show()

input('Program paused. Press enter to continue.\n')

## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.

# Try different SVM Parameters here
sigma, C = dataset3Params(X, y, Xval, yval)

model = svmTrain(X, y, C, kernelFunction = "gaussian", sigma=sigma)
visualizeBoundary(X, y, model)
plt.title('SVM Decision Boundary with sigma = %0.3f and C = %0.3f ' % (sigma, C))
plt.show()

input('Program paused. Press enter to continue.\n')
