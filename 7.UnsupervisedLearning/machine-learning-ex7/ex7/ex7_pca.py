# Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from featureNormalize import featureNormalize
from pca import pca
from projectData import *
from recoverData import *
from displayData import displayData

# ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize

#  The following command loads the dataset. You should now have the
#  variable X in your environment

data = loadmat('ex7data1.mat')
X = data['X']

#  Visualize the example dataset
fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], 'bo', fillstyle='none')
ax.set(aspect='equal', xlim=(0.5, 6.5), ylim=(2, 8))
fig.show()

input('Program paused. Press enter to continue.\n')

# =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique.

print('\nRunning PCA on example dataset.\n\n')

#  Before running PCA, it is important to first normalize X

X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, S = pca(X_norm)

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.

eigen1 = mu + 1.5 * S[0]*U[:, 0]
eigen2 = mu + 1.5 * S[1]*U[:, 1]

#  Visualize the example dataset
fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], 'bo', fillstyle='none')
ax.set(aspect='equal', xlim=(0.5, 6.5), ylim=(2, 8))
ax.plot([mu[0], eigen1[0]], [mu[1], eigen1[1]], '-k', linewidth=2)
ax.plot([mu[0], eigen2[0]], [mu[1], eigen2[1]], '-k', linewidth=2)
fig.show()

print('Top eigenvector: \n')
print(' U(:,0) = %f %f \n' % (U[0, 0], U[1, 0]))
print('\n(you should expect to see -0.707107 -0.707107)\n')

input('Program paused. Press enter to continue.\n')

# =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the
#  first k eigenvectors. The code will then plot the data in this reduced
#  dimensional space.  This will show you what the data looks like when
#  using only the corresponding eigenvectors to reconstruct it.

print('\nDimension reduction on example dataset.\n\n')

# Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: %f\n' % (Z[0]))
print('\n(this value should be about 1.481274)\n\n')

X_rec = recoverData(Z, U, K)
print('Approximation of the first example: %f %f\n' %
      (X_rec[0, 0], X_rec[0, 1]))
print('\n(this value should be about  -1.047419 -1.047419)\n\n')

# Plot the normalized dataset (returned from pca)
fig, ax = plt.subplots()
ax.plot(X_norm[:, 0], X_norm[:, 1], 'bo', fillstyle='none')
ax.set(aspect='equal', xlim=(-4, 3), ylim=(-4, 3))
#  Draw lines connecting the projected points to the original points
ax.plot(X_rec[:, 0], X_rec[:, 1], 'ro', fillstyle='none')

for i in range(np.size(X_norm, 0)):
    ax.plot([X_norm[i, 0], X_rec[i, 0]], [
            X_norm[i, 1], X_rec[i, 1]], '--k', linewidth=1)

fig.show()

input('Program paused. Press enter to continue.\n')

# =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment

print('\nLoading face dataset.\n\n')

#  Load Face dataset
data = loadmat('ex7faces.mat')
X = data['X']

# Display the first 100 faces in the dataset
frame = displayData(X[:100, :])
plt.imshow(frame, cmap='gray')
plt.axis('off')
plt.show()


print('Program paused. Press enter to continue.\n')

# =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.

print('\nRunning PCA on face dataset.\n'
         '(this might take a minute or two ...)\n\n')

# Before running PCA, it is important to first normalize X by subtracting
#  the mean value from each feature
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, S = pca(X_norm)

#  Visualize the top 36 eigenvectors found
frame = displayData(U[:, :36].transpose())
plt.imshow(frame, cmap='gray')
plt.axis('off')
plt.show()


# ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors

print('\nDimension reduction for face dataset.\n\n')

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: ')
print(Z.shape)

input('\n\nProgram paused. Press enter to continue.\n')

# ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print('\nVisualizing the projected (reduced dimension) faces.\n\n')

K = 100
X_rec = recoverData(Z, U, K)


# Creates two subplots and unpacks the output array immediately
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# Display normalized data
frame = displayData(X_norm[:100, :])
ax1.imshow(frame, cmap='gray', aspect = 'equal')
ax1.set_title('Original faces')
ax1.set_axis_off()
# Display reconstructed data from only k eigenfaces
frame = displayData(X_rec[:100, :])
ax2.imshow(frame, cmap='gray', aspect = 'equal')
ax2.set_title('Recovered faces')
ax2.set_axis_off()

fig.show()

input('Program paused. Press enter to continue.\n')


'''
%% === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
%  One useful application of PCA is to use it to visualize high-dimensional
%  data. In the last K-Means exercise you ran K-Means on 3-dimensional
%  pixel colors of an image. We first visualize this output in 3D, and then
%  apply PCA to obtain a visualization in 2D.

close all; close all; clc

% Reload the image from the previous exercise and run K-Means on it
% For this to work, you need to complete the K-Means assignment first
A = double(imread('bird_small.png'));

% If imread does not work for you, you can try instead
%   load ('bird_small.mat');

A = A / 255;
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3);
K = 16;
max_iters = 10;
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

%  Sample 1000 random indexes (since working with all the data is
%  too expensive. If you have a fast computer, you may increase this.
sel = floor(rand(1000, 1) * size(X, 1)) + 1;

%  Setup Color Palette
palette = hsv(K);
colors = palette(idx(sel), :);

%  Visualize the data and centroid memberships in 3D
figure;
scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
title('Pixel dataset plotted in 3D. Color shows centroid memberships');
fprintf('Program paused. Press enter to continue.\n');
pause;

%% === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
% Use PCA to project this cloud to 2D for visualization

% Subtract the mean to use PCA
[X_norm, mu, sigma] = featureNormalize(X);

% PCA and project the data to 2D
[U, S] = pca(X_norm);
Z = projectData(X_norm, U, 2);

% Plot in 2D
figure;
plotDataPoints(Z(sel, :), idx(sel), K);
title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');
fprintf('Program paused. Press enter to continue.\n');
pause;
'''
