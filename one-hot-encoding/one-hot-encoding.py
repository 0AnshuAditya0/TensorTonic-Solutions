import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    y = np.asarray(y)
    
    if num_classes is None:
        num_classes = np.max(y) + 1
        
    return np.eye(num_classes, dtype=int)[y].tolist()