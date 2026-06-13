import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len is None:
        if len(seqs) == 0:
            L = 0
        else:
            L = max(len(seq) for seq in seqs)
    else:
        L = max_len

    N = len(seqs)
    
    result = np.full((N, L), pad_value, dtype=np.int32)

    for i, seq in enumerate(seqs):
        if len(seq) == 0 or L == 0:
            continue
        length_to_copy = min(len(seq), L)
        result[i, :length_to_copy] = seq[:length_to_copy]
        
    return result