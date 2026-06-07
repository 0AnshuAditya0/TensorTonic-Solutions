import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pe = np.zeros((seq_len, d_model))

    position = np.arange(seq_len)[:, np.newaxis]

    for j in range(d_model):
        
        i = j // 2

        divisor = base ** (2 * i / d_model)

        angle = position / divisor

        if j % 2 == 0:
            pe[:, j] = np.sin(angle).flatten()
        else:
            pe[:, j] = np.cos(angle).flatten()
            
    return pe