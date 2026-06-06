import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    
    return np.trapezoid(tpr, fpr)