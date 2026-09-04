def get_normalized_coords_comp2(poly, img_size):
    """
    Comp2 (paper label; ablation of Baseline = coords_baseline.py, "CHAI (Ours)"): fully raw
    coordinates - NO PCA rotation, NO X-mirror, NO Y-flip, nothing but a straight axis-aligned
    bbox from the polygon's raw image-space points, normalized by image width/height. This
    isolates "what does the model get if we skip every geometric transform entirely," as opposed
    to Comp3 (coords_comp3.py - Y-flip kept, X-mirror dropped, still pooled) which only isolates
    the X-mirror step alone.

    REDEFINED 2026-08-31 (was previously identical to what's now coords_comp3.py: Y-flip kept,
    X-mirror dropped, pooled both-jaws). No jaw parameter needed any more - nothing branches on
    it, since there's no Y-flip left to decide. Because there's no transform left to align upper
    and lower jaw coordinate ranges (that WAS the Y-flip's job - see coords_baseline.py's
    docstring: upper/lower are shot with opposite vertical framing), Comp2 can no longer be a
    single pooled model - ResNet/tooth/train_comp2.py trains lower and upper as two fully
    separate models instead (same structure as Transformer/complexity's per-jaw training).

    poly: [[x, y], ...] segmentation polygon points, in original image coordinates.
    img_size: (width, height) of the image.
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
