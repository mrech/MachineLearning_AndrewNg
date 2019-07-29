# Machine Learning Online Class
#  Exercise 6 | Support Vector Machines

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from plotData import *
from sklearn import svm
from visualizeBoundaryLinear import *
from gaussianKernel import *

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

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)

PenaltyParameter = 100

model = svm.LinearSVC(tol=1e-3, C= PenaltyParameter, random_state=0, max_iter=100000)
model.fit(X,y.ravel())

visualizeBoundaryLinear(X, y, model)
plt.show()

input('Program paused. Press enter to continue.\n')

## =============== Part 3: Implementing Gaussian Kernel ===============
#  Implement the Gaussian kernel to use with the SVM.

print('\nEvaluating the Gaussian Kernel ...\n')

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

sim = gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {:.6f} :' \
         '\n\t{:.6f}\n(for sigma = 2, this value should be about 0.324652)\n'.format(float(sigma), float(sim)))

input('Program paused. Press enter to continue.\n')

'''
%% =============== Part 4: Visualizing Dataset 2 ================
%  The following code will load the next dataset into your environment and 
%  plot the data. 
%

fprintf('Loading and Visualizing Data ...\n')

% Load from ex6data2: 
% You will have X, y in your environment
load('ex6data2.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
%  After you have implemented the kernel, we can now use it to train the 
%  SVM classifier.
% 
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');

% Load from ex6data2: 
% You will have X, y in your environment
load('ex6data2.mat');

% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to
% convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Part 6: Visualizing Dataset 3 ================
%  The following code will load the next dataset into your environment and 
%  plot the data. 
%

fprintf('Loading and Visualizing Data ...\n')

% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

%  This is a different dataset that you can use to experiment with. Try
%  different values of C and sigma here.
% 

% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');

% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

'''
