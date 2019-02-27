# MATRICES AND VECTORS WITH NumPy
import numpy as np

# Create an ndarray
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# Inizialize a vector
v = np.array([[1], [2], [3]])

# Get the dimension of the matrix A where m = rows and n = columns
(m, n) = A.shape
# Get the dimansion of the vector
dim_v = v.shape

# Now let's index into the 2nd row and 3rd column of matrix A
A_23 = A[1, 2]

# ADDITION AND SCALAR MULTIPLICATION

# Initialize matrix A and B
A = np.array([[1, 2, 4], [5, 3, 2]])
B = np.array([[1, 3, 4], [1, 1, 1]])

# Inizialize a constant s
s = 2

# See how element-wise addition/subtraction works
add_AB = A + B
sub_AB = A - B

# See how scalar multiplication works
mult_As = A * s

# Divide A by s
div_As = A / s

# What happens if we have a Matrix + scalar
add_As = A + s

# MATRIX-VECTOR MULTIPLICATION
A = np.arange(1, 10).reshape(3, 3)

# Initialize vector v
v = np.ones((3, 1), dtype=np.int8)

# Multiply A * v (Matrix product)
Av = A @ v

# MATRIX-MATRIX MULTIPLICATION
A = np.arange(1, 7).reshape(3, 2)
B = np.array([[1], [2]])

# We expect a resulting matrix of (3x2)*(2x1)=(3x1)
mult_AB = A @ B

# MATRIX MULTIPLICATION PROPERTIES
# Initialize random matrices A and B
A = np.array([[1, 2], [4, 5]])
B = np.array([[1, 1], [0, 2]])

# Initialize a 2 by 2 identity matrix
I = np.identity(2, dtype=np.int8)

# I*A vs A*I results in the original matrix
AI = A @ I
IA = I @ A

# IA = AI but AB != BA
# Matrices are not commutative: A∗B≠B∗A
# Matrices are associative: (A∗B)∗C=A∗(B∗C)
AB = A @ B
BA = B @ A


# INVERSE AND TRANSPOSE
A = np.array([[1, 2, 0], [0, 5, 6], [7, 0, 9]])

# Transpose A
A_trans = np.transpose(A)

# Take the inverse of A
A_inv = np.linalg.inv(A)

# Multiplying by the inverse results in the identity matrix.
A_invA = np.linalg.inv(A) @ A
