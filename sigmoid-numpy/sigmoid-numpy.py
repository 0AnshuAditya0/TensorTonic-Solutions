import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    arr = np.asarray(x)
    res = 1 / (1 + np.exp(-arr))
    return res