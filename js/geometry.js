// True PCA over all detected teeth: mean centroid position + principal-axis angle from an
// eigen-decomposition of the centroid covariance matrix. This is the single PCA used
// everywhere on the site - the "PCA" overlay checkbox, arch ordering, FDI badges, and the
// ResNet Tooth model's meta features (computeToothMeta) - and matches functions/features/
// theta.py's calculate_pca_rotation() exactly, which is what builds the training CSVs.
const calculatePCARotation = (predictions) => {
    if (!predictions || predictions.length === 0) return null;

    const centroids = predictions
        .filter(pred => pred.polygon && pred.polygon.length > 0)
        .map(pred => computeCentroid(pred.polygon));

    const n = centroids.length;
    if (n === 0) return null;

    let meanX = 0, meanY = 0;
    centroids.forEach(c => { meanX += c.x; meanY += c.y; });
    meanX /= n;
    meanY /= n;

    let angle = 0;
    if (n >= 2) {
        let sxx = 0, syy = 0, sxy = 0;
        centroids.forEach(c => {
            const dx = c.x - meanX;
            const dy = c.y - meanY;
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
        });
        const denom = n - 1;
        const covXX = sxx / denom;
        const covYY = syy / denom;
        const covXY = sxy / denom;

        if (Math.abs(covXX) > 1e-9 || Math.abs(covYY) > 1e-9 || Math.abs(covXY) > 1e-9) {
            // Eigen-decomposition of the symmetric 2x2 matrix [[covXX, covXY], [covXY, covYY]] -
            // take the eigenvector for the larger eigenvalue (principal axis), matching
            // np.linalg.eigh + descending sort + evecs[:, 0] in the Python pipeline
            const trace = covXX + covYY;
            const det = covXX * covYY - covXY * covXY;
            const disc = Math.sqrt(Math.max(0, trace * trace - 4 * det));
            const lambda1 = (trace + disc) / 2;

            let vx, vy;
            if (Math.abs(covXY) > 1e-9) {
                vx = covXY;
                vy = lambda1 - covXX;
            } else if (covXX >= covYY) {
                vx = 1; vy = 0;
            } else {
                vx = 0; vy = 1;
            }

            angle = Math.atan2(vy, vx);
            if (angle > Math.PI / 2) angle -= Math.PI;
            else if (angle < -Math.PI / 2) angle += Math.PI;
        }
    }

    // For the on-canvas overlay line, project every centroid onto the PCA axis and use the
    // extreme projections as the line's two endpoints - purely visual, doesn't affect the
    // center/angle used everywhere else
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    let minProj = 0, maxProj = 0;
    centroids.forEach(c => {
        const proj = (c.x - meanX) * cosA + (c.y - meanY) * sinA;
        if (proj < minProj) minProj = proj;
        if (proj > maxProj) maxProj = proj;
    });
    const pLeft = { x: meanX + minProj * cosA, y: meanY + minProj * sinA };
    const pRight = { x: meanX + maxProj * cosA, y: meanY + maxProj * sinA };

    return {
        pLeft,
        pRight,
        center: { x: meanX, y: meanY },
        angle
    };
};

// Rotates a point into the PCA-aligned coordinate frame (centered on pca.center)
const rotateToPcaFrame = (x, y, pca) => {
    if (!pca) return { tx: x, ty: y };
    const cos = Math.cos(-pca.angle);
    const sin = Math.sin(-pca.angle);
    const rx = x - pca.center.x;
    const ry = y - pca.center.y;
    return {
        tx: rx * cos - ry * sin,
        ty: rx * sin + ry * cos
    };
};

// Determines jaw (upper/lower) purely from the dental arch's curvature: fits a parabola
// y = a*x^2 + b*x + c directly to the tooth centroids' RAW (un-rotated) image coordinates and
// checks the sign of the leading coefficient. Deliberately does NOT rotate into the PCA frame
// first (unlike this function's original version) - leave-one-out testing (removing each
// posterior molar in turn, the highest-leverage points for PCA's axis estimate) showed 0%
// misclassification for both the PCA-rotated and raw-coordinate fits, identically, across 3,000
// sampled arches per jaw - so the rotation step was earning nothing here. Also validated against
// this project's full GT set (71,522 train+val arches): lower-jaw arches always fit with
// a < 0, upper-jaw arches always fit with a > 0, with a clean margin between the two ranges
// (lower tops out at -0.00088, upper starts at +0.00098) - so this fully replaces the
// ResNet/jaw CNN classifier and its /classify round trip. Needs >= 4 centroids for a
// non-degenerate quadratic fit. `pca` param kept (unused) for call-site compatibility.
const classifyJawByCurvature = (predictions, pca) => {
    if (!predictions) return null;
    const centroids = predictions
        .filter(pred => pred.polygon && pred.polygon.length > 0)
        .map(pred => computeCentroid(pred.polygon));
    if (centroids.length < 4) return null;

    const pts = centroids.map(c => ({ tx: c.x, ty: c.y }));

    // Least-squares fit via the 3x3 normal-equations system for ry = a*rx^2 + b*rx + c.
    let s1 = 0, s2 = 0, s3 = 0, s4 = 0, t0 = 0, t1 = 0, t2 = 0;
    const n = pts.length;
    pts.forEach(({ tx, ty }) => {
        const x2 = tx * tx;
        s1 += tx; s2 += x2; s3 += x2 * tx; s4 += x2 * x2;
        t0 += ty; t1 += tx * ty; t2 += x2 * ty;
    });

    // Solve [[s4,s3,s2],[s3,s2,s1],[s2,s1,n]] * [a,b,c]^T = [t2,t1,t0]^T via Cramer's rule
    const det3 = (m) => (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    );
    const detM = det3([[s4, s3, s2], [s3, s2, s1], [s2, s1, n]]);
    if (Math.abs(detM) < 1e-12) return null; // degenerate (e.g. centroids collinear)

    const a = det3([[t2, s3, s2], [t1, s2, s1], [t0, s1, n]]) / detM;
    const b = det3([[s4, t2, s2], [s3, t1, s1], [s2, t0, n]]) / detM;
    const c = det3([[s4, s3, t2], [s3, s2, t1], [s2, s1, t0]]) / detM;

    return { isUpper: a > 0, a, b, c };
};

// The FDI quadrant (tens digit) only depends on jaw (upper/lower) + which side of the image's
// own horizontal center (width/2) a tooth sits on - unlike slot-based lookup, this works even
// when the tooth count doesn't reconstruct to 12, so it's used for the Holding-state FDI badges
// (ResNet fallback). offsetFromCenter is the tooth's raw x minus width/2 (see computeHoldingTens
// in js/render.js) - was PCA-rotated x before 2026-08-31, switched to width/2 for consistency
// with computeToothMetaPooled (see that function's docstring for the accuracy trade-off).
const computeQuadrantTens = (offsetFromCenter, isUpper) => {
    const isLeftSide = offsetFromCenter < 0;
    if (isUpper) return isLeftSide ? 1 : 2;
    return isLeftSide ? 4 : 3;
};

// Computes the same normalized [x1, y1, x2, y2, theta] independent variables the ResNet Tooth
// model was trained on (see functions/features/coords.py + theta.py) from a live prediction's
// polygon, using the shared PCA from calculatePCARotation(). Mirroring is decided purely
// by which side of the PCA centerline the tooth's centroid falls on (positive rotated-x = the
// mirrored FDI quadrant), matching the rule used to build the training CSVs - so this works
// without knowing the tooth's FDI number up front.
const computeToothMeta = (pred, pca) => {
    const centroid = computeCentroid(pred.polygon);
    const { tx: ctx, ty: cty } = rotateToPcaFrame(centroid.x, centroid.y, pca);
    const mirror = ctx >= 0;

    // Rotate every polygon point into the PCA-aligned frame and re-fit an axis-aligned box
    // from the rotated points - rotating only the box's two corners would turn a rectangle
    // into a parallelogram, so this has to match coords.py's get_normalized_coords exactly.
    let x1_c = Infinity, x2_c = -Infinity, y1_c = Infinity, y2_c = -Infinity;
    pred.polygon.forEach(pt => {
        const { tx, ty } = rotateToPcaFrame(pt[0], pt[1], pca);
        if (tx < x1_c) x1_c = tx;
        if (tx > x2_c) x2_c = tx;
        if (ty < y1_c) y1_c = ty;
        if (ty > y2_c) y2_c = ty;
    });

    if (mirror) {
        const x1_new = -x1_c;
        const x2_new = -x2_c;
        x1_c = Math.min(x1_new, x2_new);
        x2_c = Math.max(x1_new, x2_new);
    }

    let theta = Math.atan2(cty, ctx) / Math.PI;
    if (mirror) {
        theta = theta >= 0 ? 1.0 - theta : -1.0 - theta;
    }

    return {
        // Normalized against the source image's own dimensions (currentImage), not canvas.width/
        // height - the canvas is padded taller than the image by BOTTOM_PANEL_HEIGHT to make room
        // for drawBottomPanel's info strip (js/main.js), so canvas.height no longer equals the
        // image height these features were trained against (functions/features/coords.py).
        x1: x1_c / currentImage.width,
        y1: y1_c / currentImage.height,
        x2: x2_c / currentImage.width,
        y2: y2_c / currentImage.height,
        theta,
        // Which quadrant side this tooth is on (same test as computeQuadrantTens) - sent to
        // server.py /tooth_predict so it can group teeth by quadrant for the duplicate-digit
        // post-processing, without needing to know the (yet-to-be-predicted) FDI number itself.
        mirror
    };
};

// Mirrored Variant counterpart to computeToothMeta: computes the [x1, y1, x2, y2] independent
// variables the retired Mirrored Variant ResNet Tooth model was trained on (see
// functions/features/coords_mirrored.py + ResNet/tooth/train_mirrored.py) - no remaining call
// sites as of the CHAI codebase rename (js/api.js runToothAnalysis now uses
// computeToothMetaComplexity instead, CHAI's own unmirrored convention). Left in place rather
// than deleted, same rollback-safety rationale as server.py's unused per-jaw tooth_models dict.
// computeToothMeta itself is left untouched because js/render.js's Arch Complexity call
// (drawFdiNumbers) still needs its old PCA-rotated output (the Transformer/complexity model's own
// training data still uses that per-jaw PCA-rotated coordinate scheme - see that model's
// docstring - it just dropped the theta column, not the rotation).
//
// Two things differ from computeToothMeta: no PCA rotation at all (the bbox is taken directly
// from the raw polygon, and the mirror/flip axes are the image's own geometry - width/2, height -
// instead of a per-arch PCA axis), both found to be more robust to missing teeth than rotating
// the whole frame (see project notes). `mirror` (which side of the image's own horizontal center
// a tooth sits on) also switched to width/2 as of 2026-08-31 (was PCA-rotated centroid sign
// before - PCA was found to be ~0.03-0.04pp more accurate on a held-out missing-teeth test, but
// width/2 was chosen anyway so nothing in this pipeline depends on the PCA axis at all - see
// functions/graph/tens_norot.py for the Python-side equivalent used in evaluation). No `pca`
// param needed any more as a result.
const computeToothMetaPooled = (pred, isUpper) => {
    const centroid = computeCentroid(pred.polygon);
    const mirror = centroid.x >= currentImage.width / 2;

    let x1_c = Infinity, x2_c = -Infinity, y1_c = Infinity, y2_c = -Infinity;
    pred.polygon.forEach(pt => {
        if (pt[0] < x1_c) x1_c = pt[0];
        if (pt[0] > x2_c) x2_c = pt[0];
        if (pt[1] < y1_c) y1_c = pt[1];
        if (pt[1] > y2_c) y2_c = pt[1];
    });

    const width = currentImage.width;
    const height = currentImage.height;

    if (mirror) {
        const x1_new = width - x2_c;
        const x2_new = width - x1_c;
        x1_c = x1_new;
        x2_c = x2_new;
    }

    if (!isUpper) {
        const y1_new = height - y2_c;
        const y2_new = height - y1_c;
        y1_c = y1_new;
        y2_c = y2_new;
    }

    return {
        x1: x1_c / width,
        y1: y1_c / height,
        x2: x2_c / width,
        y2: y2_c / height,
        mirror
    };
};

// Shared by the Arch Complexity Transformer and CHAI's own tooth-number classifier - both use the
// identical coordinate scheme (see functions/features/coords_chai.py's
// get_normalized_coords_chai, which both Transformer/complexity/build_dataset.py's cache and
// js/api.js runToothAnalysis's /tooth_predict payload are built from): no PCA rotation, no
// X-mirror (unlike the retired Mirrored Variant's computeToothMetaPooled - both tasks need the
// true whole-arch left-right shape / quadrant signal, which X-mirror would fold away), Y-flip
// kept for the whole lower jaw (jaw-frame alignment, matching coords_chai.py's Y flip). Replaces
// the old computeToothMeta(pred, pca) call the Complexity model used before its coordinate-scheme
// swap. computeToothMeta itself now has NO remaining call sites anywhere (the "Coords Mirroring"
// debug overlay, js/render.js drawMirroringOverlay, only ever duplicated its PCA mirror-decision
// logic inline, never actually called it) - left in place rather than deleted, same
// rollback-safety rationale as server.py's unused per-jaw tooth_models dict.
const computeToothMetaComplexity = (pred, isUpper) => {
    let x1_c = Infinity, x2_c = -Infinity, y1_c = Infinity, y2_c = -Infinity;
    pred.polygon.forEach(pt => {
        if (pt[0] < x1_c) x1_c = pt[0];
        if (pt[0] > x2_c) x2_c = pt[0];
        if (pt[1] < y1_c) y1_c = pt[1];
        if (pt[1] > y2_c) y2_c = pt[1];
    });

    const width = currentImage.width;
    const height = currentImage.height;

    if (!isUpper) {
        const y1_new = height - y2_c;
        const y2_new = height - y1_c;
        y1_c = y1_new;
        y2_c = y2_new;
    }

    return {
        x1: x1_c / width,
        y1: y1_c / height,
        x2: x2_c / width,
        y2: y2_c / height,
    };
};

const computeCentroid = (polygon) => {
    if (!polygon || polygon.length === 0) return { x: 0, y: 0 };
    let sumX = 0;
    let sumY = 0;
    polygon.forEach(pt => {
        sumX += pt[0];
        sumY += pt[1];
    });
    return { x: sumX / polygon.length, y: sumY / polygon.length };
};

// Orders teeth by their PCA-rotated x coordinate (left-to-right along the arch's main axis).
// An earlier version instead solved for the shortest Hamiltonian path via Held-Karp dynamic
// programming; an ablation study (see the paper's Supplementary Material) found this added no
// measurable benefit to the pipeline's final tens-digit/quadrant accuracy - the downstream
// digit-occurrence-order correction step only needs the coarse left/right split to be correct,
// and this plain sort gets that right just as often, at a fraction of the cost. `isUpper` is
// unused now (kept for call-site compatibility) - the rotated-x sign convention is the same for
// both jaws.
const computeArchOrder = (vertices, pca, isUpper) => {
    const n = vertices.length;
    if (n === 0) return [];

    const getRotatedX = (pt) => {
        if (!pca) return pt.x;
        const cos = Math.cos(-pca.angle);
        const sin = Math.sin(-pca.angle);
        return (pt.x - pca.center.x) * cos - (pt.y - pca.center.y) * sin;
    };

    return vertices
        .map((pt, i) => ({ i, rx: getRotatedX(pt) }))
        .sort((a, b) => a.rx - b.rx)
        .map(({ i }) => i);
};

