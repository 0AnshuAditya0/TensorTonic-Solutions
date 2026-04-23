import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    
    w = np.array(w, dtype=np.float64)
    g = np.array(g, dtype=np.float64)
    G = np.array(G, dtype=np.float64)
    
    G += g**2
    w -= (lr / np.sqrt(G + eps)) * g
    
    return w.tolist(), G.tolist()