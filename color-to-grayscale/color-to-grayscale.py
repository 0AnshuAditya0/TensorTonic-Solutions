def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    grayscale_image = []
    
    for row in image:
        new_row = []
        for pixel in row:
            r, g, b = pixel[0], pixel[1], pixel[2]
            
            gray_value = (0.299 * r) + (0.587 * g) + (0.114 * b)
            
            new_row.append(round(gray_value, 3))
            
        grayscale_image.append(new_row)
        
    return grayscale_image