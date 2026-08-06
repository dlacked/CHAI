import math

def rotate_points(points, mean_pt, angle):
    """
    Rotates a list of [x, y] points into the PCA-aligned frame: translate by -mean_pt, then
    rotate by -angle. Matches the transform theta.py's get_theta_values() applies to centroids.
    """
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    rotated = []
    for x, y in points:
        rx = x - mean_pt[0]
        ry = y - mean_pt[1]
        tx = rx * cos_a - ry * sin_a
        ty = rx * sin_a + ry * cos_a
        rotated.append((tx, ty))
    return rotated

def get_normalized_coords(poly, mean_pt, angle, img_size, fdi_number):
    """
    Rotates the tooth's segmentation polygon into the PCA-aligned frame (centered on mean_pt,
    rotated by -angle - the same transform used for theta), takes the axis-aligned bounding box
    of the rotated points, mirrors x for FDI Q2 & Q3, and normalizes by image dimensions.

    Rotating only the original box's two corners would NOT be correct here: rotating a
    rectangle turns it into a parallelogram, so a valid axis-aligned box in the rotated frame
    has to be re-fit from the polygon's boundary points, not just two diagonal corners.
    Args:
        poly (list): [[x, y], ...] segmentation polygon points.
        mean_pt (list or tuple): [mean_x, mean_y] PCA center point.
        angle (float): PCA rotation angle (radians).
        img_size (tuple): (width, height) of the image.
        fdi_number (int): FDI tooth number (e.g. 31, 46).
    Returns:
        tuple: (x1_norm, y1_norm, x2_norm, y2_norm) as formatted strings.
    """
    if not poly or mean_pt is None or img_size is None:
        return "0.0000", "0.0000", "0.0000", "0.0000"

    width, height = img_size
    if width <= 0 or height <= 0:
        return "0.0000", "0.0000", "0.0000", "0.0000"

    # 1. Rotate the polygon into the PCA-aligned frame and take its axis-aligned bounding box
    rotated = rotate_points(poly, mean_pt, angle)
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    x1_c, x2_c = min(xs), max(xs)
    y1_c, y2_c = min(ys), max(ys)

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
