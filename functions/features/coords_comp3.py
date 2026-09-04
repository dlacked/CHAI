def get_normalized_coords_comp3(poly, img_size, jaw):
    """
    coords_baseline.py minus its X-mirror step - dedicated coordinate computation for Comp3
    (paper label; up-to-tens-digit, 12-way, no post-process), an ablation of Baseline (=
    coords_baseline.py, "CHAI (Ours)"). No fdi_number parameter needed - without mirroring,
    nothing branches on which quadrant a tooth is in.

    RENAMED 2026-08-31 from coords_comp2.py: this used to also serve Comp2 (last-digit-only,
    6-way), back when Comp2 was defined as "Baseline minus X-mirror, still pooled via this same
    Y-flip". Comp2 was then redefined to fully-raw coordinates (no transform at all, incl. no
    Y-flip) with separate per-jaw models instead of pooling - see functions/features/coords_comp2.py
    (the NEW file at that name) - so this file (and csv_comp3/) is now Comp3-only.

    Keeps Baseline's Y flip (flip_y = jaw == "lower") - that's jaw-frame alignment (upper/lower
    are separate photos shot with opposite vertical framing, see coords_baseline.py's docstring),
    a different thing from the Q2/Q3 X-mirror this drops. Y-flip is harmless to keep for Comp3
    (unlike X-mirror, it's a uniform whole-arch reflection, not a selective fold of one quadrant
    onto another, so it doesn't erase any geometric information Comp3 needs) and IS needed here
    since Comp3 stays a single pooled (both-jaws) model. Dropping the X-mirror is deliberate:
    Comp3 needs the quadrant signal these un-mirrored coordinates still carry (mirrored
    coordinates erase it structurally - the whole point of mirroring is making both quadrants
    land on the same coordinate) in order to predict the tens digit at all.

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
