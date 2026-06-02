import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    vocab_indices = {word: i for i, word in enumerate(vocab)}

    vector = np.zeros(len(vocab), dtype=int)

    for token in tokens:
        if token in vocab_indices:
            vector[vocab_indices[token]] += 1
            
    return vector