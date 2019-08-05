# VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM


def visualizeBoundary(X, y, model):

   '''
   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
   boundary learned by the SVM and overlays the data on it
   '''

   import numpy as np
   from plotData import plotData
   import matplotlib.pyplot as plt
   from sklearn import svm
   from gaussianKernelGramMatrix import gaussianKernelGramMatrix

   # Make classification predictions over a grid of values
   # Meshgrid return coordinate matrices from coordinate vectors
   x_min, x_max = X[:, 0].min(), X[:, 0].max()
   y_min, y_max = X[:, 1].min(), X[:, 1].max()
   xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
   
   Z = model.predict(gaussianKernelGramMatrix(np.c_[xx.ravel(), yy.ravel()], X))

   # alternative without the Gaussian Matrix implementation
   #Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

   Z = Z.reshape(xx.shape)
   
   plotData(X,y)
   plt.contour(xx, yy, Z, colors = 'b', levels = [0.5])