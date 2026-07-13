import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    r1 = np.array(rater1)
    r2 = np.array(rater2)

    classes = np.unique(np.concatenate((r1, r2)))
    n_classes = len(classes)
    n_samples = len(r1)
    
    if n_samples == 0:
        return 0.0

    class_to_idx = {val: idx for idx, val in enumerate(classes)}

    cm = np.zeros((n_classes, n_classes), dtype=float)
    for i in range(n_samples):
        idx1 = class_to_idx[r1[i]]
        idx2 = class_to_idx[r2[i]]
        cm[idx1, idx2] += 1

    p_o = np.trace(cm) / n_samples

    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    p_e = np.sum(row_sums * col_sums) / (n_samples ** 2)
    
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
        
    return float((p_o - p_e) / (1.0 - p_e))