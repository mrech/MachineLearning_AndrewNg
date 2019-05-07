import numpy as np
import math
import matplotlib.pyplot as plt


def displayData(X):
    '''
        DISPLAYDATA Display 2D data in a nice grid
        Input nxn pixel grayscale images.
    Where nxn pixel are represented as column and each image as row.
        Returns a grid with all the images.
    '''

    # Identify the nxn pixel
    image_width = int(np.sqrt(X.shape[1]))
    [m, n] = X.shape
    image_height = int(n / image_width)

    # Compute number of rows and columns to display on the grid
    display_rows = math.floor(np.sqrt(m))
    display_cols = math.ceil(m/display_rows)

    # Setup blank display
    _, axes = plt.subplots(display_rows, display_cols)
    curr_img = 0

    # Assign each image into the subplots
    for i in range(display_rows):
        for j in range(display_cols):
            axes[i, j].imshow(X[curr_img].reshape((image_width, image_height),
                                                  order='F'),
                              cmap='gray')
            axes[i, j].axis('off') # Turn the axis off
            curr_img += 1

    # Display grid
    plt.show()

