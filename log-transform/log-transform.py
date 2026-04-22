import math

def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    a = np.array(values)
    return np.log(a+1)