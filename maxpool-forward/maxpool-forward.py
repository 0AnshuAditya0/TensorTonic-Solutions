def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    H = len(X)
    W = len(X[0]) if H > 0 else 0

    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    
    output = []

    for r in range(out_H):
        row = []
        for c in range(out_W):
            start_r = r * stride
            start_c = c * stride

            window_vals = [
                X[i][j] 
                for i in range(start_r, start_r + pool_size) 
                for j in range(start_c, start_c + pool_size)
            ]

            row.append(max(window_vals))
            
        output.append(row)
        
    return output