import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x, dtype=float)
    
    if rng is None:
        rng = np.random.default_rng()

    if p == 0.0:
        mask = np.ones_like(x)
        return (x, (mask * 1.0))
        
    keep_prob = 1.0 - p
    binary_mask = rng.binomial(1, keep_prob, size=x.shape)

    scale_factor = 1.0 / keep_prob
    dropout_pattern = binary_mask * scale_factor

    output = x * dropout_pattern
    
    return output, dropout_pattern
