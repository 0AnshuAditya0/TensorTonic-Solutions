def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    if not values:
        return []
        
    min_v = min(values)
    max_v = max(values)
    val_range = max_v - min_v
    
    if val_range == 0:
        return [0] * len(values)

    width = val_range / num_bins
    
    result = []
    for v in values:
        bin_idx = int((v - min_v) // width)
        
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
            
        result.append(bin_idx)
        
    return result