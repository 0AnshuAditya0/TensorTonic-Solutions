import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    unique, counts = np.unique(y_train, return_counts=True)
    majority = unique[np.argmax(counts)]
    
    return np.full(len(X_test), majority).tolist()