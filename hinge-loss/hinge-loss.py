import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    a = np.array(y_true)
    b = np.array(y_score)

    m = np.maximum(0,margin-a*b)

    if reduction=="mean":
        return np.mean(m)
    else:
        return np.sum(m)

    return m