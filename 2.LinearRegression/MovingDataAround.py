import numpy as np
import os
import pandas as pd

# matrices
A = np.arange(1, 7).reshape(3, 2)
A.shape

# vector
v = np.arange(1, 5)
len(v)

# Current working directory
os.getcwd()
os.chdir('/home/morena/MachineLearning/AndrewNg_Python/Dataset/')

# read file
dataset = pd.read_csv('Data.csv')

dir()  # in scope variables
globals()  # dictionary of global variables
locals()  # dictionary of local variables

dataset.shape

# subsetting
v = dataset[0:3]
del v

# Manipulate data
A[2, 1]
A[1, :]  # ':' means every element along that row/column
A[:, 1]
A[(0, 2), :]  # get everything from the first and third row
A[:, 1] = [10, 11, 12]  # Assignement

y = np.array([[100], [101], [102]])
A = np.append(A, y, axis=1)  # append another column vector to right
A.shape

# put all elements of A into a single vector
np.array(A).reshape(-1, order='F')
help(np.reshape)

# Concatenate matrices
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[11, 12], [13, 14], [15, 16]])
C = np.append(A, B, axis=1)
C = np.append(A, B, axis=0)
