import itertools
def interaction_features(X):
    """
    Generate pairwise interaction features and append them to the original features.
    """
    if not X or not X[0]:
        return X
        
    n_features = len(X[0])
    
    pair_indices = list(itertools.combinations(range(n_features), 2))
    
    result = []
    for row in X:
        
        interactions = [row[i] * row[j] for i, j in pair_indices]
        
        result.append(row + interactions)
        
    return result
