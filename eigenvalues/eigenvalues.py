import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    if not matrix or not isinstance(matrix, list):
        return None

    num_rows = len(matrix)
    for row in matrix:
        if not isinstance(row, list) or len(row) != num_rows:
            return None

    A = np.array(matrix)
        
    eigenvalues = np.linalg.eigvals(A)
    eigenvalues_sorted = np.sort(eigenvalues)
    
    return np.array(eigenvalues_sorted)