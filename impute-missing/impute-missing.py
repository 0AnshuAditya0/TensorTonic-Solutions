import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    X_arr = np.array(X, dtype=float)
    
    is_1d = X_arr.ndim == 1
    if is_1d:
        X_arr = X_arr.reshape(-1, 1) 

    for col_idx in range(X_arr.shape[1]):
        col = X_arr[:, col_idx]
        nan_mask = np.isnan(col)

        if np.all(nan_mask):
            fill_val = 0.0
        else:
            
            valid_vals = col[~nan_mask]
            if strategy == 'mean':
                fill_val = np.mean(valid_vals)
            elif strategy == 'median':
                fill_val = np.median(valid_vals)
            else:
                raise ValueError("Strategy must be 'mean' or 'median'")
        col[nan_mask] = fill_val
        
    if is_1d:
        return X_arr.flatten().tolist()
    return X_arr.tolist()