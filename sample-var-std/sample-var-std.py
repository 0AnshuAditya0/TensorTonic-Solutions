import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    a = np.array(x)
    var = np.var(x, ddof=1)
    dev = np.sqrt(var)
    return var, dev