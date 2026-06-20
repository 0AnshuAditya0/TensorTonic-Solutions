import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
