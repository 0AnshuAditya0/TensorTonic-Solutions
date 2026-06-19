import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)
    
    d = np.sqrt(np.sum((a - b) ** 2, axis=-1))

    loss_similar = y * (d ** 2)

    loss_dissimilar = (1 - y) * (np.maximum(0, margin - d) ** 2)

    total_loss = loss_similar + loss_dissimilar

    if reduction == "mean":
        return float(np.mean(total_loss))
    elif reduction == "sum":
        return float(np.sum(total_loss))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")