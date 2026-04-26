import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    a = np.array(y_pred)
    b = np.array(y_true)

    return np.sum(np.square(a-b))/a.size
