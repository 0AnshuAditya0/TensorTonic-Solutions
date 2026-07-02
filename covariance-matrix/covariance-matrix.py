import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.array(X)

    if X.ndim < 2 or X.shape[0] < 2:
        return None

    cov_matrix = np.cov(X, rowvar=False)

    if cov_matrix.ndim == 0:
        return cov_matrix.reshape(1, 1)
        
    return cov_matrix