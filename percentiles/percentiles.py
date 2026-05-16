import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    a = np.array(x)
    b = np.array(q)

    return np.percentile(a, b, method='linear')