import numpy as np
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    
    a = np.array(x)
    return np.where( a> 0, a, alpha*(np.exp(a) - 1)).tolist()