import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    if x.shape != p.shape:
        raise ValueError(f"x and p must have same shape: {x.shape} vs {p.shape}")
    
    if len(x) == 0:
        raise ValueError("x and p cannot be empty")
    
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities must be between 0 and 1")
    
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError(f"probabilities must sum to 1, got {np.sum(p)}")
    
    return np.sum(x * p)
