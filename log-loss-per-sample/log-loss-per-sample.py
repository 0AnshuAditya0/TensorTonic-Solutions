import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    losses = []
    for y, p in zip(y_true, y_pred):
        cl = max(eps, min(1 - eps, p))

        if y == 1:
            loss = -math.log(cl)
        else:
            loss = -math.log(1 - cl)
            
        losses.append(loss)
    return losses