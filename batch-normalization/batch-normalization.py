import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.array(x, dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta = np.array(beta, dtype=float)
    
    if x.ndim == 2:
        
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        x_hat = (x - mean) / np.sqrt(var + eps)
        
        out = gamma * x_hat + beta
        
    elif x.ndim == 4:
        
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + eps)

        gamma_reshaped = gamma.reshape(1, -1, 1, 1)
        beta_reshaped = beta.reshape(1, -1, 1, 1)

        out = gamma_reshaped * x_hat + beta_reshaped
        
    else:
        raise ValueError("This function only supports 2D or 4D inputs.")
        
    return out.tolist()