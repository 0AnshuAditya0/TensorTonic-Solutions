import numpy as np
def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    arr = np.asarray(values)
    
   
    ranks_low = np.argsort(np.argsort(arr)) + 1

    ranks_high = len(arr) - np.argsort(np.argsort(-arr))

    average_ranks = (ranks_low + ranks_high) / 2.0
    
    return average_ranks.tolist()