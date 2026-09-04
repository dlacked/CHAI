def get_normalized_coords_baseline(poly, img_size, fdi_number, jaw):
    """
    This is Baseline (paper label) - the main CHAI model, "CHAI (Ours)" in the paper tables.
    coords_unified_mirror.py / main_unified_mirror.py / train_unified_mirror.py were the original
    attempt at this (back when it was called variant C), but were superseded before ever being
    trained (no model_unified_mirror/ was ever produced): that version kept PCA rotation and had
    an X-mirror bug (mirrored tens=4 when it shouldn't have - see this module's own docstring
    below for the corrected rule). This _pooled version fixes both. All three superseded files,
    along with variant A (main.py) and B (coords_nomirror.py/main_nomirror.py/train_nomirror.py,
    dropped from the paper), have been moved to backups/superseded_20260831/.

    coords.get_normalized_coords (moved to backups/) was the ablation counterpart this started
    from for the pooled (both-jaws, one model) setup: NO PCA rotation at all - the axis-aligned
    bbox is taken directly from the polygon's
    raw image-space coordinates, then normalized by image width/height. PCA rotation was found to
    be more harmful than helpful (see project notes): the correction it applies for real camera
    tilt is small (~2deg average on complete arches) while a single missing posterior molar can
    swing the estimated axis by ~13deg, corrupting every tooth's coordinates in that arch, not
    just the missing one's neighbor.

    Two independent mirror/flip operations replace it, both using only information that doesn't
    depend on which teeth happen to be present:
      - X mirror (mirror_x = tens in (2, 3)): same rule production already uses per-jaw - NOT
        tens in (2, 3, 4). Verified empirically per-quadrant (raw x-range of the digit-1 tooth,
        which sits closest to the midline, across ~800 samples per quadrant): FDI 11 (Q1, ref)
        ~[0.362, 0.500], FDI 41 (Q4) ~[0.410, 0.503] - already on the SAME side as Q1 in their
        own (separate) raw images, no X mirror needed, only the Y flip below. FDI 21 (Q2)
        ~[0.501, 0.638] and FDI 31 (Q3) ~[0.500, 0.594] both sit on the opposite side and need the
        X mirror (confirmed: mirroring 21's range lands on ~[0.362, 0.499], matching Q1 almost
        exactly). Reflected around the image's own horizontal center (width/2), not a
        tooth-derived centroid - the capture protocol frames the arch centered in the photo, so
        this is robust to missing teeth by construction (see project notes for why mean_pt-based
        centering was rejected).
      - Y flip (flip_y = jaw == "lower"): upper and lower jaw photos are separate images shot
        with opposite framing conventions (looking up vs. looking down), so the same tens digit's
        vertical position is inverted between them - confirmed empirically (FDI 16 raw y-range
        ~[0.68, 0.97] vs FDI 46 ~[0.04, 0.37], which is a near-exact match once flipped:
        1-[0.04,0.37] = [0.63, 0.96]). Without this, a pooled model would see the "same" quadrant
        in wildly different vertical positions depending on jaw.

    poly: [[x, y], ...] segmentation polygon points, in original image coordinates.
    img_size: (width, height) of the image.
    fdi_number: FDI tooth number (e.g. 31, 46) - decides the X mirror only (tens 2/3/4 vs 1).
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

    if fdi_number != -1 and fdi_number is not None:
        tens = fdi_number // 10
        if tens in (2, 3):
            x1_new = width - x2_c
            x2_new = width - x1_c
            x1_c, x2_c = x1_new, x2_new

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
