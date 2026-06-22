def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    rows = len(X)
    cols = len(X[0])
    
    output = []

    for r in range(0, rows, pool_size):
        output_row = []

        for c in range(0, cols, pool_size):
            total_sum = 0
            
            for pr in range(r, r + pool_size):
                for pc in range(c, c + pool_size):
                    total_sum += X[pr][pc]

            pool_area = pool_size * pool_size
            output_row.append(total_sum / pool_area)

        output.append(output_row)
        
    return output