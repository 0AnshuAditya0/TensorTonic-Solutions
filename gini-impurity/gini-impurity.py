import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    y_left = np.array(y_left)
    y_right = np.array(y_right)
    
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    
    if n_total == 0:
        return 0.0
        
    def calculate_gini(y):
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / n
        return 1.0 - np.sum(probabilities ** 2)
        
    gini_left = calculate_gini(y_left)
    gini_right = calculate_gini(y_right)
    
    return (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    