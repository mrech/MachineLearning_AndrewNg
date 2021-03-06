# Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from cofiCostFunc import cofiCostFunc
from checkCostFunction import checkCostFunction
from loadMovieList import loadMovieList
from normalizeRatings import normalizeRatings
from scipy import optimize

# =============== Part 1: Loading movie ratings dataset ================
#  You will start by loading the movie ratings dataset to understand the
#  structure of the data.

print('Loading movie ratings dataset.\n\n')

data = loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on
#  943 users

#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): %f / 5\n\n' %
      np.mean(Y[0, R[0, :]]))

#  We can "visualize" the ratings matrix by plotting it with imagesc

plt.imshow(Y)
plt.ylabel('Movies')
plt.xlabel('Users')
# make a color bar
plt.colorbar(boundaries=[0, 1, 2, 3, 4, 5], ticks=[0, 1, 2, 3, 4, 5])
plt.show()

print('\nProgram paused. Press enter to continue.\n')

# ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in
#  cofiCostFunc.m to return J.

#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
data = loadmat('ex8_movieParams.mat')
X = data['X']
Theta = data['Theta']

#  Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3

X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]

# unroll the parameters into a single vector params
params = []
params.extend((list(X.flatten(order='F')) +
               list(Theta.flatten(order='F'))))

#  Evaluate cost function
J, _ = cofiCostFunc(params, Y, R, num_users, num_movies,
                    num_features, 0)

print('Cost at loaded parameters: %f '
      '\n(this value should be about 22.22)\n' % (J))

input('\nProgram paused. Press enter to continue.\n')

# ============== Part 3: Collaborative Filtering Gradient ==============
#  Once your cost function matches up with ours, you should now implement
#  the collaborative filtering gradient function. Specifically, you should
#  complete the code in cofiCostFunc.m to return the grad argument.

print('\nChecking Gradients (without regularization) ... \n')

#  Check gradients by running checkNNGradients
checkCostFunction()

input('\nProgram paused. Press enter to continue.\n')

# ========= Part 4: Collaborative Filtering Cost Regularization ========
#  Now, you should implement regularization for the cost function for
#  collaborative filtering. You can implement it by adding the cost of
#  regularization to the original cost computation.

#  Evaluate cost function
J, _ = cofiCostFunc(params, Y, R, num_users, num_movies,
                    num_features, 1.5)

print('Cost at loaded parameters (lambda = 1.5): %f '
      '\n(this value should be about 31.34)\n' % (J))

input('\nProgram paused. Press enter to continue.\n')

# ======= Part 5: Collaborative Filtering Gradient Regularization ======
#  Once your cost matches up with ours, you should proceed to implement
#  regularization for the gradient.


print('\nChecking Gradients (with regularization) ... \n')

# Check gradients by running checkCostFunction
checkCostFunction(1.5)

input('\nProgram paused. Press enter to continue.\n')

# ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed. This
#  part of the code will also allow you to put in your own ratings for the
#  movies in our dataset!

movieList = loadMovieList()

#  Initialize my ratings
my_ratings = np.zeros((1682, 1))

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set

my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

print('\n\nNew user ratings:\n')

for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for %s' % (my_ratings[i], movieList.get(i)))

input('\nProgram paused. Press enter to continue.\n')

# ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating
#  dataset of 1682 movies and 943 users

print('\nTraining collaborative filtering...\n')

#  Load data
data = loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by
#  943 users

#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  Add our own ratings to the data matrix
Y = np.hstack((my_ratings, Y))
R = np.hstack((my_ratings != 0, R))

#  Normalize Ratings
Ynorm, Ymean = normalizeRatings(Y, R)

#  Useful Values
num_users = np.size(Y, 1)
num_movies = np.size(Y, 0)
num_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters = []
initial_parameters.extend((list(X.flatten(order='F')) +
                           list(Theta.flatten(order='F'))))

# Set options for using advanced optimizer

# Set Regularization
lambda_par = 10

myargs = (Ynorm, R, num_users, num_movies, num_features, lambda_par)

theta = optimize.minimize(cofiCostFunc, initial_parameters,
                          args=myargs, method='CG', jac=True,
                          options={'disp': True, 'maxiter': 500})

# Unfold the returned theta back into U and W
X = np.reshape(theta['x'][:num_movies*num_features],
               (num_movies, num_features), order='F')

Theta = np.reshape(theta['x'][num_movies*num_features:],
                (num_users, num_features), order = 'F')

print('Recommender system learning completed.\n')

input('\nProgram paused. Press enter to continue.\n')

## ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.

p = np.dot(X, Theta.T) + Ymean
my_predictions = p[:,0]


ix = np.argsort(my_predictions)[::-1]

print('\nTop recommendations for you:\n')

for i in range(10):
    j = ix[i]
    print('Predicting rating %.1f for movie %s'%( my_predictions[j],
            movieList.get(j)))
            
print('\n\nOriginal ratings provided:\n')

for i in range(len(my_ratings)):
      if my_ratings[i] > 0:
            print('Rated %d for %s' % (my_ratings[i], movieList.get(i)))

input('\nProgram paused. Press enter to continue.\n')

# Find related movies to Usual Suspects, The (1995)
similar = []
for i in range(np.size(X,0)):
      current = np.sum(np.abs(X[11, ] - X[i,]))
      similar.append(current)

print('\nMovies similar to Usual Suspects, The (1995): ')
idx = np.argsort(similar)
for i in range(10):
      ax = idx[i]
      print(movieList.get(ax))

