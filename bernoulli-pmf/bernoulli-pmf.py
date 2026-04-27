import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    a = np.array(x)
    b = np.array(p)

    pmf = np.where(a == 1, p, np.where(a == 0, 1 - p, 0))
    return pmf, p, p*(1-p)