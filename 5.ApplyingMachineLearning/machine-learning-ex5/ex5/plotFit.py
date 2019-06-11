# PLOTFIT Plots a learned polynomial regression fit over an existing figure.
# Also works with linear regression.

def plotFit(min_x, max_x, mu, sigma, theta, p):
    '''
    PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
    fit with power p and feature normalization (mu, sigma).
    '''

    import numpy as np
    from polyFeatures import polyFeatures
    import matplotlib.pyplot as plt

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x-15, max_x + 25, 0.05)

    # Map the X values 
    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly/sigma

    # Add ones
    X_poly = np.hstack((np.ones((X_poly.shape[0],1)), X_poly))

    # Plot
    plt.plot(x, X_poly.dot(theta), '--', linewidth=2)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')