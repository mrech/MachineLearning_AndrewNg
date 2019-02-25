
# Basic Operations
import math
import numpy as np
import matplotlib.pyplot as plt

a = math.pi
print(f'2 dicimals: {a:.2f}')
print(f'2 dicimals: {a:.6f}')

# Generate vector
v = np.linspace(1,2, num=11,endpoint=True)
v = np.arange(1,7)

# Generate matrices
np.ones((2,3))
C = np.full((2,3), 2)
W = np.zeros((1,3))
# random number drawn from uniform distribution from 0 and 1
w = np.random.rand(3)
W = np.random.rand(3,3)
#random values from a gaussian distribution (mean = 0, sd = 1)
w = np.random.randn(3)

# Histogram Using Matplotlib
w = -6 + math.sqrt(10)*(np.random.randn(10000))
plt.hist(w)
plt.hist(w, 50)
plt.show()

# Identity matrix
I = np.identity(4)
help(np.identity)