import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    p = np.array(p)
    y = np.array(y)

    p_t = np.where(y == 1, p, 1 - p)

    loss = - ((1 - p_t) ** gamma) * np.log(p_t)
    
    return np.mean(loss)