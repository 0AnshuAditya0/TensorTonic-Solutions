import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    if k < 0:
        return 0.0, 0.0

    steps = np.arange(k + 1)
    
    f = np.array([np.prod(np.arange(1, i + 1)) for i in steps])

    all_pmfs = (lam**steps * np.exp(-lam)) / f
    
    pmf = all_pmfs[-1]     
    cdf = np.sum(all_pmfs)  
    
    return pmf, cdf