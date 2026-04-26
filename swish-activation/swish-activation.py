import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    a = np.array(x)

    s = 1/(1+np.exp(-a))
    return a*s