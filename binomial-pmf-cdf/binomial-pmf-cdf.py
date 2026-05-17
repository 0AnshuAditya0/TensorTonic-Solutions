import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    p = float(p)
    q = 1.0 - p
    
    pmf = float(comb(n, k, exact=False) * (p**k) * (q**(n - k)))
    
    cdf = 0.0
    for i in range(int(k) + 1):
        cdf += comb(n, i, exact=False) * (p**i) * (q**(n - i))
    cdf = float(cdf)
    
    return pmf, cdf