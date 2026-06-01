import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)
    
    if y.size == 0:
        return 0.0
        
    _, counts = np.unique(y, return_counts=True)

    probabilities = counts / len(y)
    
    probabilities = probabilities[probabilities > 0]

    return float(-np.sum(probabilities * np.log2(probabilities)))