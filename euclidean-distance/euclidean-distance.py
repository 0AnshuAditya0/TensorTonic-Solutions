import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    a = np.array(x)
    b = np.array(y)

    return np.linalg.norm(a - b)