import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    matrix = np.array(matrix, dtype=float)
    
    if matrix.ndim != 2:
        return None

    if axis is not None and axis not in (0, 1):
        return None
    if norm_type not in ('l1', 'l2', 'max'):
        return None
    
    if norm_type == 'l1':
        norms = np.sum(np.abs(matrix), axis=axis, keepdims=True)
    elif norm_type == 'l2':
        norms = np.sqrt(np.sum(np.square(matrix), axis=axis, keepdims=True))
    elif norm_type == 'max':
        norms = np.max(np.abs(matrix), axis=axis, keepdims=True)
        
    norms = np.where(norms == 0, 1.0, norms)
    
    return matrix / norms