def get_coords(box):
    """
    Format bounding box coordinates to 4 decimal places.
    Args:
        box (list): [x1, y1, x2, y2] bounding box.
    Returns:
        tuple: (x1, y1, x2, y2) as formatted strings.
    """
    if not box or len(box) < 4:
        return "0.0000", "0.0000", "0.0000", "0.0000"
    return (
        f"{box[0]:.4f}",
        f"{box[1]:.4f}",
        f"{box[2]:.4f}",
        f"{box[3]:.4f}"
    )

def get_normalized_coords(box, mean_pt, img_size, fdi_number):
    """
    Center the bounding box coordinates relative to the PCA center point (mean_pt),
    apply mirroring (x sign flip) for FDI Q2 & Q3, and normalize by image dimensions.
    Args:
        box (list): [x1, y1, x2, y2] bounding box.
        mean_pt (list or tuple): [mean_x, mean_y] PCA center point.
        img_size (tuple): (width, height) of the image.
        fdi_number (int): FDI tooth number (e.g. 31, 46).
    Returns:
        tuple: (x1_norm, y1_norm, x2_norm, y2_norm) as formatted strings.
    """
    if box is None or len(box) < 4 or mean_pt is None or img_size is None:
        return "0.0000", "0.0000", "0.0000", "0.0000"
        
    width, height = img_size
    if width <= 0 or height <= 0:
        return "0.0000", "0.0000", "0.0000", "0.0000"
        
    # 1. Centering (Relative to PCA center point)
    x1_c = box[0] - mean_pt[0]
    y1_c = box[1] - mean_pt[1]
    x2_c = box[2] - mean_pt[0]
    y2_c = box[3] - mean_pt[1]
    
    # 2. Invert x sign if FDI tens digit is 2 or 3 (Quadrant 2 or 3)
    if fdi_number != -1 and fdi_number is not None:
        tens = fdi_number // 10
        if tens in (2, 3):
            # Invert signs
            x1_new = -x1_c
            x2_new = -x2_c
            # Keep x1 as min and x2 as max
            x1_c = min(x1_new, x2_new)
            x2_c = max(x1_new, x2_new)
            
    # 3. Normalization (Divide by width & height)
    x1_norm = x1_c / width
    y1_norm = y1_c / height
    x2_norm = x2_c / width
    y2_norm = y2_c / height
    
    return (
        f"{x1_norm:.4f}",
        f"{y1_norm:.4f}",
        f"{x2_norm:.4f}",
        f"{y2_norm:.4f}"
    )
