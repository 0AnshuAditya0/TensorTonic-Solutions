import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    x_arr = np.asarray(x, dtype=float)
    h_arr = np.asarray(h_prev, dtype=float)
    
    feat_x = x_arr.shape[-1]
    feat_h = h_arr.shape[-1]
    X, x_was_1d = _as2d(x, feat_x)
    H_prev, h_was_1d = _as2d(h_prev, feat_h)

    Wz = np.asarray(params.get('Wz', params.get('W_z', np.zeros((feat_x, feat_h)))), dtype=float)
    Wr = np.asarray(params.get('Wr', params.get('W_r', np.zeros((feat_x, feat_h)))), dtype=float)
    Wh = np.asarray(params.get('Wh', params.get('W_h', np.zeros((feat_x, feat_h)))), dtype=float)
    
    Uz = np.asarray(params.get('Uz', params.get('U_z', np.zeros((feat_h, feat_h)))), dtype=float)
    Ur = np.asarray(params.get('Ur', params.get('U_r', np.zeros((feat_h, feat_h)))), dtype=float)
    Uh = np.asarray(params.get('Uh', params.get('U_h', np.zeros((feat_h, feat_h)))), dtype=float)
    
    bz = np.asarray(params.get('bz', params.get('b_z', np.zeros(feat_h))), dtype=float)
    br = np.asarray(params.get('br', params.get('b_r', np.zeros(feat_h))), dtype=float)
    bh = np.asarray(params.get('bh', params.get('b_h', np.zeros(feat_h))), dtype=float)

    if Wz.shape == (feat_h, feat_x): Wz = Wz.T
    if Wr.shape == (feat_h, feat_x): Wr = Wr.T
    if Wh.shape == (feat_h, feat_x): Wh = Wh.T

    z = _sigmoid(np.dot(X, Wz) + np.dot(H_prev, Uz) + bz)       
    r = _sigmoid(np.dot(X, Wr) + np.dot(H_prev, Ur) + br)       
    h_tilde = np.tanh(np.dot(X, Wh) + np.dot(r * H_prev, Uh) + bh) 

    H_next = (1 - z) * H_prev + z * h_tilde
    
    if x_was_1d or h_was_1d:
        return H_next.flatten()
        
    return H_next