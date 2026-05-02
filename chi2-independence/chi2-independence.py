import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """

    a = np.array(C)
    
    row_sums = a.sum(axis=1)
    col_sums = a.sum(axis=0)
    total = a.sum()

    expected = np.outer(row_sums, col_sums) / total

    chi2 = np.sum((a - expected)**2 / expected)
    
    return chi2, expected
    