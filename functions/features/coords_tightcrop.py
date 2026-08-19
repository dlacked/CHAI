import math

from coords import rotate_points

# Same padding (in PCA-rotated-frame pixels) js/render.js's TIGHT_CROP_PAD used before it was
# reverted for the ablation - kept in sync manually since this is Python, not shared code.
TIGHT_CROP_PAD = 100


def compute_arch_tight_bbox(polys_with_fdi, mean_pt, angle, pad=TIGHT_CROP_PAD):
    """
    The union bounding box of every tooth's polygon in ONE arch, in the PCA-rotated frame -
    i.e. what js/render.js's computePcaAlignedBBox computes for the "Tight Crop" visualization,
    just as numbers instead of a canvas crop, with one deliberate difference: the X extent is
    computed in the SAME mirrored (Q2/Q3-flipped) frame get_normalized_coords_tightcrop puts
    each tooth's own X in, not the raw unmirrored photo. Mirroring folds the two original
    quadrants of an arch onto one canonical side (see coords.get_normalized_coords) - taking the
    union bbox of the RAW coordinates would mix that folded frame with an unfolded crop width,
    which stretches every tooth's normalized X inconsistently depending on which quadrant it
    happened to start in. Y is unaffected (only X ever gets mirrored), so it's still the plain
    union of the raw rotated Y extents.

    polys_with_fdi: list of (poly, fdi_number) - poly is [[x, y], ...] in original image
    coordinates, fdi_number decides whether that tooth's X gets the same Q2/Q3 mirror
    get_normalized_coords_tightcrop will apply to it individually. mean_pt, angle: the arch's
    shared PCA (calculate_pca_rotation). One crop bbox is shared by every tooth in the arch.
    Returns (tx1, ty1, tx2, ty2) - the crop's bounds in the (mirror-consistent) rotated frame,
    pad already applied.
    """
    tx1 = ty1 = float("inf")
    tx2 = ty2 = float("-inf")
    for poly, fdi_number in polys_with_fdi:
        rotated = rotate_points(poly, mean_pt, angle)
        xs = [p[0] for p in rotated]
        ys = [p[1] for p in rotated]
        x1_c, x2_c = min(xs), max(xs)

        if fdi_number is not None and fdi_number != -1 and (fdi_number // 10) in (2, 3):
            x1_c, x2_c = -x2_c, -x1_c

        if x1_c < tx1:
            tx1 = x1_c
        if x2_c > tx2:
            tx2 = x2_c
        for ty in ys:
            if ty < ty1:
                ty1 = ty
            if ty > ty2:
                ty2 = ty
    return tx1 - pad, ty1 - pad, tx2 + pad, ty2 + pad


def get_normalized_coords_tightcrop(poly, mean_pt, angle, crop_bbox, fdi_number):
    """
    coords.get_normalized_coords's ablation counterpart: same PCA-rotate + re-fit-bbox-from-
    polygon + Q2/Q3 mirror steps, but normalized against the ARCH's own tight crop bounds
    (crop_bbox, from compute_arch_tight_bbox - shared by every tooth in this arch) instead of
    the full source photo's width/height. This is the whole point of the ablation: the same
    physical tooth position should land at the same normalized coordinate regardless of how
    much empty margin the photographer happened to leave around the arch when framing the shot -
    see the project memory/discussion this pairs with for the full reasoning and the known
    trade-off (crop bounds now depend on which teeth got detected, a new source of train/
    inference variation that coords.py's full-image normalization never had).

    poly, mean_pt, angle, fdi_number: same as coords.get_normalized_coords.
    crop_bbox: (tx1, ty1, tx2, ty2) from compute_arch_tight_bbox for this tooth's arch.
    Returns (x1_norm, y1_norm, x2_norm, y2_norm) as formatted strings.
    """
    if not poly or mean_pt is None or crop_bbox is None:
        return "0.0000", "0.0000", "0.0000", "0.0000"

    tx1, ty1, tx2, ty2 = crop_bbox
    crop_w = tx2 - tx1
    crop_h = ty2 - ty1
    if crop_w <= 0 or crop_h <= 0:
        return "0.0000", "0.0000", "0.0000", "0.0000"

    # 1. Rotate the polygon into the PCA-aligned frame and take its axis-aligned bounding box -
    # identical to coords.get_normalized_coords's own first step.
    rotated = rotate_points(poly, mean_pt, angle)
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    x1_c, x2_c = min(xs), max(xs)
    y1_c, y2_c = min(ys), max(ys)

    # 2. Invert x sign if FDI tens digit is 2 or 3 (Quadrant 2 or 3) - identical to
    # coords.get_normalized_coords.
    if fdi_number != -1 and fdi_number is not None:
        tens = fdi_number // 10
        if tens in (2, 3):
            x1_new = -x1_c
            x2_new = -x2_c
            x1_c = min(x1_new, x2_new)
            x2_c = max(x1_new, x2_new)

    # 3. Normalize against the arch's own tight crop bounds instead of the full image - the one
    # step that differs from coords.get_normalized_coords. crop_bbox must come from
    # compute_arch_tight_bbox (whose X extent is already computed in this same mirrored frame),
    # not a raw unmirrored union bbox, or a Q2/Q3 tooth's mirrored x1_c/x2_c would get normalized
    # against a crop window that was never mirrored to match it.
    x1_norm = (x1_c - tx1) / crop_w
    y1_norm = (y1_c - ty1) / crop_h
    x2_norm = (x2_c - tx1) / crop_w
    y2_norm = (y2_c - ty1) / crop_h

    return (
        f"{x1_norm:.4f}",
        f"{y1_norm:.4f}",
        f"{x2_norm:.4f}",
        f"{y2_norm:.4f}"
    )
