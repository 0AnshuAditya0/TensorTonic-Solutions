import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    a = np.array(x, dtype=np.float64)

    return np.where(a > 0, a, alpha * a)
