# Display 20 by 20 grid of pixels “unrolled” into a
# 400-dimensional vector


def dispalyData(X):
    '''
    dispalyData(X): displays one 20x20 pixels imagage
    'unrolled into a 400-dimensional vector'
    '''

    import matplotlib.pyplot as plt

    # randomly select a row from X, which represent 20x20pixel image
    plt.imshow(X.reshape((20, 20), order='F'), cmap='gray')
    plt.axis('off')

    return plt.show()
