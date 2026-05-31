import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """

    dot = np.dot(a, b)
    
    n = np.linalg.norm(a)
    m = np.linalg.norm(b)
    
    if n == 0 or m == 0:
        return 0.0
        
    return float(dot / (n * m))
    