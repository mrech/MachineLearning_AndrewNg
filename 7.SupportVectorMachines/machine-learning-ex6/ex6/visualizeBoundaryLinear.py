# %VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the SVM

def visualizeBoundaryLinear(X, y, model):
    '''
    VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
    learned by the SVM and overlays the data on it
    '''

    import numpy as np
    from plotData import plotData
    import matplotlib.pyplot as plt

    b = model.intercept_
    w = model.coef_
    xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), num=50)
    # Calculate the decision boundary line: 
    # g(z) = 1/2 >> e^(-z) = 1 >> z = 0 
    # theta0 + theta1X1 + theta2X2 = 0
    # x2 plays as y >> y = - (theta0 + theta1X1) / theta2
    yp = -(w.item(0)*xp + b)/w.item(1)
    plotData(X,y)
    plt.plot(xp,yp,'-b')
