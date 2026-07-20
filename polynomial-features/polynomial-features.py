import numpy as np
def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    return np.vander(values, N=degree + 1, increasing=True).tolist()