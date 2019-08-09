# DISPLAYDATA Display 2D data in a nice grid


def displayData(X):
    '''
    [DISPLAYDATA(X) creates the frame which displays multiple 2D image.
    '''

    import matplotlib.pyplot as plt
    import numpy as np

    m, n = np.shape(X)

    picture_width = int(np.sqrt(n))
    picture_height = int((n / picture_width))

    rows_picture = int(np.floor(np.sqrt(m)))
    columns_picture = int(np.floor(m / rows_picture))

    frame = np.zeros((rows_picture*picture_height,
                     columns_picture*picture_width))

    img = 0
    for i in range(rows_picture):  # goes to the picture in the rows
        for j in range(columns_picture):  # goes to the picture in the columns
            rows_step = i*picture_height
            columns_step = j*picture_width
            frame[(rows_step):(rows_step+picture_height),
                    (columns_step):(columns_step+picture_width)] = \
                        X[img].reshape(
                            (picture_height, picture_width), order='F')
            img += 1
            
    return frame


    


