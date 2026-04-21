import numpy as np
def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    a = np.array(box_a, dtype=float)
    b = np.array(box_b, dtype=float)
    
    x1 = max(a[0], b[0])  
    y1 = max(a[1], b[1])  
    x2 = min(a[2], b[2])  
    y2 = min(a[3], b[3])  

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    
    union_area = area_a + area_b - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area