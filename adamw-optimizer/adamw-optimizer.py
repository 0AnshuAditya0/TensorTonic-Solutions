import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    w = np.array(w, dtype=np.float64)
    m = np.array(m, dtype=np.float64)
    v = np.array(v, dtype=np.float64)
    grad = np.array(grad, dtype=np.float64)
    
    m_new = beta1 * m + (1 - beta1) * grad
    
    v_new = beta2 * v + (1 - beta2) * (grad**2)

    w_decayed = w - lr * weight_decay * w
    
    w_new = w_decayed - lr * m_new / (np.sqrt(v_new) + eps)

    return (np.round(w_new, 5)), (np.round(m_new, 5)), (np.round(v_new, 5))