import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g_arr = np.array(g, dtype=float)
    
    if max_norm <= 0:
        return g_arr
        
    total_norm = np.linalg.norm(g_arr)
    
    if total_norm > max_norm:
        g_arr *= (max_norm / total_norm)
        
    return g_arr