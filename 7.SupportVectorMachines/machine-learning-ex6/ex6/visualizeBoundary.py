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
   # Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

   Z = Z.reshape(xx.shape)
   
   plotData(X,y)
   plt.contour(xx, yy, Z, colors = 'b', levels = [0.5])


   


'''

   for i in range(X1.shape[1]):
      this_X = np.column_stack((X1[:, i], X2[:, i]))
      vals[:, i] = model.decision_function(this_X) 

   return vals

   # Plot the SVM boundary
   plt.contour(X1, X2, vals, colors="b", levels=[0,0])
   # Plot the training data.
   plotData(X, y)


   Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
   Z = Z.reshape(xx.shape)

   plt.contour(xx, yy, Z, levels = [0])
   # Plot the samples
   plotData(X, y)
   
   # Plot decision boundary
   #plt.contour(xx, yy, Z, levels = [0.5, 1])

   X1, X2 = np.meshgrid(x1plot, x2plot)

   vals = np.zeros(X1.shape)
   for i in range(X1.shape[1]):
      this_X = [X1[:, i], X2[:, i]]
      vals[:, i] = model.predict(this_X)

   #vals = model.predict(np.c_[X1.ravel(), X2.ravel()])
   #vals = vals.reshape(X1.shape)

   #plotData(X, y)
   #plt.contour(X1, X2, vals)

   return vals


xx, yy = np.meshgrid(np.linspace(0, 2, num = 1),\
                     np.linspace(np.min(X[:,1]), np.max(X[:,1]), num = 100))




   vals = np.zeros(X1.shape)
   for i in range(X1.shape[1]):
      this_X = np.c_[X1[:, i], X2[:, i]]
      vals[:, i] = model.predict(this_X)

   return vals
   

    , varargin
      svm.SVC.predict(model, this_X)

   # Plot the SVM boundary
   plot_
   contour(X1, X2, vals, [0.5 0.5], 'b');

X1, X2 = np.meshgrid(np.linspace(0, 2, num=3),
                        np.linspace(0, 2, num=3))
this_X = np.column_stack((X1[:, 0], X2[:, 0]))
'''