import numpy as np
def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    ref = np.array(reference_counts, dtype=float)
    prod = np.array(production_counts, dtype=float)
    
    ref_norm = ref / np.sum(ref)
    prod_norm = prod / np.sum(prod)
    
    tvd = 0.5 * np.sum(np.abs(ref_norm - prod_norm))
    
    return {
        "score": float(tvd),
        "drift_detected": bool(tvd > threshold)
    }