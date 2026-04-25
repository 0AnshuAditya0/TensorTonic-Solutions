import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    a = np.array(A)
    r, c = np.indices((a.shape[1], a.shape[0]))
    # r,c = c, r
    return a[c, r]
