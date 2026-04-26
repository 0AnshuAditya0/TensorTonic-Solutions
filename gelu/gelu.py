import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    a = np.array(x)
    vec_erf = np.vectorize(math.erf)
    
    return 0.5 * a * (1 + vec_erf(a / np.sqrt(2)))
