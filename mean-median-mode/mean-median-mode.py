import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    a = np.array(x)
    mean = np.mean(a)
    median = np.median(a)

    c = Counter(a)
    mode = c.most_common(1)[0][0]

    return mean, median, mode