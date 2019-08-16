# VISUALIZEFIT Visualize the dataset and its estimated distribution.

def visualizeFit(X, mu, sigma2):
    '''
    VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the 
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
    '''

    import numpy as np
    from multivariateGaussian import multivariateGaussian
    import matplotlib.pyplot as plt

    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariateGaussian(np.vstack((X1.flatten(), X2.flatten())).T, mu, sigma2)
    Z = np.reshape(Z, np.shape(X1))

    fig, ax = plt.subplots()
    plt.plot(X[:,0], X[:,1], 'bx')
    plt.contour(X1, X2, Z, 10.**np.arange(-20, 0, 3))
    ax.set_xlim(0,30)
    ax.set_ylim(0,30)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')



