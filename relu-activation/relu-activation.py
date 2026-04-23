import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    a = np.asarray(x)
    
    return np.maximum(0,a)