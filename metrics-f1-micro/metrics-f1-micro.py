def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0
        
    t = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    
    return float(t / len(y_true))