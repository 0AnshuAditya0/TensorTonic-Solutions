def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    encoded = []
    
    for val in values:
        angle = 2 * math.pi * val / period
        
        sin_val = math.sin(angle)
        cos_val = math.cos(angle)
        
    
        encoded.append([sin_val, cos_val])
        
    return encoded