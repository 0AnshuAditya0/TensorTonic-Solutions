import numpy as np

def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    """
    rec_counts = np.array(item_counts)[recommendations]
    
    probabilities = rec_counts / n_users
    
    self_info = -np.log2(probabilities)

    return np.mean(self_info)