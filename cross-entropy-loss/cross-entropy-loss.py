import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    a = np.array(y_true)
    b = np.array(y_pred)
    
    c = b[np.arange(len(a)), a]

    losses = -np.log(c)

    return np.mean(losses)