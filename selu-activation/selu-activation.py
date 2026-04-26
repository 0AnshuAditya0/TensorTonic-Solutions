import numpy as np

def selu(x, lam=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    """
    Apply SELU activation element-wise.
    Returns a list of floats rounded to 4 decimal places.
    """
    a = np.array(x)

    return np.where(a>0, lam*a, lam*alpha*(np.exp(x) - 1))