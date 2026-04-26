import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    a = np.array(x)
    e = np.exp(a - np.max(a, axis=-1, keepdims = True))
    return e / e.sum(axis=-1, keepdims=True)