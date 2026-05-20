def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    hits = 0
    total = len(recommendations)
    
    if total == 0:
        return 0.0

    for rec, gt in zip(recommendations, ground_truth):
        top = set(rec[:k])
        
        if any(item in top for item in gt):
            hits += 1
            
    return hits / total