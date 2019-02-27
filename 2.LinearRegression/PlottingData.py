# Plotting Data
import numpy as np
import matplotlib.pyplot as plt
import os

t = np.arange(0, 0.98, 0.01)
y1 = np.sin(2*np.pi*4*t)  # sine

plt.plot(t, y1)
plt.show()

y2 = np.cos(2*np.pi*4*t)  # cosine

# Plot multiple graphs on the same figure
plt.plot(t, y1, label='sin')
plt.plot(t, y2, 'r', label='cos')
plt.xlabel('time')
plt.ylabel('value')
plt.legend()
plt.title('My plot')
plt.show()

# Save plot in a specific directory
os.getcwd()
os.chdir('/home/morena/MachineLearning/AndrewNg_Python/2.LinearRegression')
plt.savefig('My_plot.png')

# Plots in multi windows
plt.figure(1)
plt.plot(t, y1, label='sin')
plt.figure(2)
plt.plot(t, y2, 'r', label='cos')
plt.show()

# subplots: devides plot a 1x2 grid, access first element
plt.subplot(1, 2, 1)
plt.plot(t, y1, label='sin')
plt.subplot(1, 2, 2)
plt.plot(t, y2, 'r', label='cos')
plt.axis([0.5, 1, -1, 1])
plt.show()
plt.clf()

# Plot a mxn grid of color, where the different colors correspond to the different
# values in the matrix
A = np.random.randint(10, size=(5, 5))
plt.imshow(A, cmap='gray')
plt.colorbar()
plt.show()
