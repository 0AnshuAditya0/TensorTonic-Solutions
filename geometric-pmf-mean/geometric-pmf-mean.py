import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    k_arr = np.array(k)
    
    pmf = p * ((1 - p) ** (k_arr - 1))

    mean = 1.0 / p
    
    return pmf, mean