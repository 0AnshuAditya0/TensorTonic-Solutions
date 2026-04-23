import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    if x.ndim < 3:
        raise ValueError(f"Input must be at least 3D, got {x.ndim}D.")
    
    return np.mean(x, axis=(-2, -1))

