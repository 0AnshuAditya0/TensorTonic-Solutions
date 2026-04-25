import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    a = np.array(x)
    b = np.array(y)

    return np.dot(a,b)