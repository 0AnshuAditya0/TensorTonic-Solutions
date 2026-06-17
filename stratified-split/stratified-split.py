import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    X = np.array(X)
    y = np.array(y)
    
    if rng is None or isinstance(rng, int):
        rng = np.random.RandomState(rng)
        
    train_indices = []
    test_indices = []
    
    classes = np.unique(y)
    
    for c in classes:
        
        class_indices = np.where(y == c)[0]

        rng.shuffle(class_indices)

        n_test = int(round(len(class_indices) * test_size))

        if n_test > 0:
            class_test = class_indices[:n_test]
            class_train = class_indices[n_test:]
        else:
            class_test = class_indices[:0]
            class_train = class_indices
            
        train_indices.extend(class_train)
        test_indices.extend(class_test)
    train_indices.sort()
    test_indices.sort()

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]