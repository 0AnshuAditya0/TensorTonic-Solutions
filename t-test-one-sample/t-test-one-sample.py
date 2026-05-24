import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    x = np.array(x)
    
    n = len(x)
    sample_mean = np.mean(x)
    sample_std = np.std(x, ddof=1)

    t_stat = (sample_mean - mu0) / (sample_std / np.sqrt(n))
    
    return t_stat