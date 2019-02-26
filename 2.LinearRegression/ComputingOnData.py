# Computing on Data
import numpy as np

A = np.arange(1, 7).reshape(3, 2)
B = np.arange(11, 17).reshape(3, 2)
C = np.array([[1, 1], [2, 2]])

# Matrix manipulation
A@C

# Element-wise multiplication
A*B
# Element-wise squaring
A**2

v = np.array([[1], [2], [3]])
# Element-wise reciprocal
1/v
1/A
np.log(v)
np.exp(v)
np.abs(v)
-v  # -1*v

# take v and increment each of its elementss by one
v + np.ones((len(v), 1))
v + 1

# Transpose
np.transpose(A)

# Matrix operations
a = np.array([1, 15, 2, 0.5])
val = max(a)
[val, ind] = [max(a), np.argmax(v)]

# Conditions and Indexes
# vector
a < 3
np.where(a < 3)
np.argwhere(a < 3)
# matrix
A = np.random.rand(3, 3)
np.argwhere(A < 0.5)
[r, c] = np.where(A < 0.5)

# Operations on all elements
np.prod(a)
np.sum(a)
np.floor(a)  # round down
np.ceil(a)  # round up to the nearest int

# element-wise max of random matrix
m1 = np.random.rand(3, 3)
m2 = np.random.rand(3, 3)
np.maximum(m1, m2)
np.amax(A, axis=0)  # Max along the first axis (rows)
np.amax(A, axis=1)  # Max along the second axis (columns)
np.max(A)  # find the max element in the entire matrix

# Row, column and diagonal operations
A = np.random.randint(10, size=(3, 3))
np.sum(A, axis=0)
np.sum(A, axis=1)
# sum the diagonal
I = np.identity(3)
A*I  # wipe out everything in A, except for the diagonal entries
np.sum(A*I)
# flip diagonal up-down
I = np.flip(I, axis=1)
np.sum(A*I)

# pseudo-inverse of a matrix
temp = np.linalg.pinv(A)
temp @ A # return the identiy matrix

