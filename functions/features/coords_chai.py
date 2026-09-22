def get_normalized_coords_chai(poly, img_size, jaw):
    """
    Coordinate computation for CHAI (the official model, formerly internally called "Comp3"):
    unmirrored, normalized bounding-box coordinates for the unified 12-way (local-quadrant x
    last-digit) classifier. No fdi_number parameter needed - without mirroring, nothing branches
    on which quadrant a tooth is in.

    Keeps a Y-flip (flip_y = jaw == "lower") for jaw-frame alignment (upper/lower are separate
    photos shot with opposite vertical framing) - a uniform whole-arch reflection, not the
    selective per-quadrant X-mirror used by the retired Mirrored Variant (see
    coords_mirrored.py). CHAI needs the quadrant signal these un-mirrored coordinates carry
    (mirrored coordinates erase it structurally - the whole point of mirroring was making both
    quadrants land on the same coordinate) in order to predict the tens digit at all.

    poly: [[x, y], ...] segmentation polygon points, in original image coordinates.
    img_size: (width, height) of the image.
    jaw: "upper" or "lower" - decides the Y flip only.
    Returns (x1_norm, y1_norm, x2_norm, y2_norm) as formatted strings.
    """
    if not poly or img_size is None:
        return "0.0000", "0.0000", "0.0000", "0.0000"

    width, height = img_size
    if width <= 0 or height <= 0:
        return "0.0000", "0.0000", "0.0000", "0.0000"

    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1_c, x2_c = min(xs), max(xs)
    y1_c, y2_c = min(ys), max(ys)

    if jaw == "lower":
        y1_new = height - y2_c
        y2_new = height - y1_c
        y1_c, y2_c = y1_new, y2_new

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
