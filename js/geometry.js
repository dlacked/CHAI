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

    return { isUpper: a > 0, a };
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

// Baseline-model counterpart to computeToothMeta: computes the [x1, y1, x2, y2] independent
// variables the Baseline ResNet Tooth model was trained on (see
// functions/features/coords_baseline.py + ResNet/tooth/train_baseline.py) - used ONLY by the
// tooth-number prediction path (js/api.js runToothAnalysis -> /tooth_predict). computeToothMeta
// itself is left untouched because js/render.js's Arch Complexity call (drawFdiNumbers) still
// needs its old PCA-rotated output (the Transformer/complexity model's own training data still
// uses that per-jaw PCA-rotated coordinate scheme - see that model's docstring - it just dropped
// the theta column, not the rotation).
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

// Arch Complexity Transformer's coordinate scheme (see functions/features/coords_comp3.py's
// get_normalized_coords_comp3, which Transformer/complexity/build_dataset.py's cache is built
// from as of 2026-08-31): no PCA rotation, no X-mirror (unlike computeToothMetaPooled - the
// Complexity task needs the true whole-arch left-right shape, which X-mirror would fold away),
// Y-flip kept for the whole lower jaw (jaw-frame alignment, matching coords_baseline.py's Y flip
// - needed so Transformer/complexity/train_pooled.py's pooled model sees one consistent
// convention). Replaces the old computeToothMeta(pred, pca) call this model used before the
// coordinate-scheme swap. computeToothMeta itself now has NO remaining call sites anywhere (the
// "Coords Mirroring" debug overlay, js/render.js drawMirroringOverlay, only ever duplicated its
// PCA mirror-decision logic inline, never actually called it) - left in place rather than
// deleted, same rollback-safety rationale as server.py's unused per-jaw tooth_models dict.
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

// Builds the shortest possible path through all teeth via Held-Karp dynamic programming,
// starting from the leftmost tooth. Unlike nearest-neighbor greedy, this finds the globally
// optimal path rather than a locally greedy one - for points strung along a single curve like
// a dental arch, the shortest path reliably follows the curve's true order, even in edge cases
// where greedy can leave a straggler tooth that needs a long final jump. With at most ~15-20
// teeth the O(2^n * n^2) cost is trivial (well under a millisecond).
const computeArchOrder = (vertices, pca, isUpper) => {
    const n = vertices.length;
    if (n === 0) return [];

    const getRotatedX = (pt) => {
        if (!pca) return pt.x;
        const cos = Math.cos(-pca.angle);
        const sin = Math.sin(-pca.angle);
        return (pt.x - pca.center.x) * cos - (pt.y - pca.center.y) * sin;
    };

    const getRotatedY = (pt) => {
        if (!pca) return pt.y;
        const cos = Math.cos(-pca.angle);
        const sin = Math.sin(-pca.angle);
        return (pt.x - pca.center.x) * sin + (pt.y - pca.center.y) * cos;
    };

    let startIdx = 0;
    let targetVal = isUpper ? -Infinity : Infinity;

    vertices.forEach((pt, i) => {
        const rx = getRotatedX(pt);
        const ry = getRotatedY(pt);
        if (rx < 0) { // Left side of the arch
            if (isUpper) {
                if (ry > targetVal) {
                    targetVal = ry;
                    startIdx = i;
                }
            } else {
                if (ry < targetVal) {
                    targetVal = ry;
                    startIdx = i;
                }
            }
        }
    });

    // Fallback if no vertex is on the left side or if targetVal was not updated
    if (targetVal === -Infinity || targetVal === Infinity) {
        let minRotX = Infinity;
        vertices.forEach((pt, i) => {
            const rx = getRotatedX(pt);
            if (rx < minRotX) {
                minRotX = rx;
                startIdx = i;
            }
        });
    }

    if (n === 1) return [startIdx];

    // Precompute pairwise distances between every pair of teeth
    const dist = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            const dx = vertices[i].x - vertices[j].x;
            const dy = vertices[i].y - vertices[j].y;
            const d = Math.sqrt(dx * dx + dy * dy);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    // dp[mask][j] = shortest path that visits exactly the teeth in `mask` (always including
    // startIdx) and ends at tooth j. parent[mask][j] records the tooth visited right before j.
    const size = 1 << n;
    const dp = Array.from({ length: size }, () => new Array(n).fill(Infinity));
    const parent = Array.from({ length: size }, () => new Array(n).fill(-1));

    const startMask = 1 << startIdx;
    dp[startMask][startIdx] = 0;

    for (let mask = 0; mask < size; mask++) {
        if (!(mask & startMask)) continue; // every path must include the starting tooth
        for (let j = 0; j < n; j++) {
            if (!(mask & (1 << j))) continue;
            const costToJ = dp[mask][j];
            if (costToJ === Infinity) continue;

            for (let k = 0; k < n; k++) {
                if (mask & (1 << k)) continue; // already visited
                const nextMask = mask | (1 << k);
                const newCost = costToJ + dist[j][k];
                if (newCost < dp[nextMask][k]) {
                    dp[nextMask][k] = newCost;
                    parent[nextMask][k] = j;
                }
            }
        }
    }

    // Pick whichever tooth minimizes total path length once every tooth has been visited
    const fullMask = size - 1;
    let bestEnd = startIdx;
    let bestCost = Infinity;
    for (let j = 0; j < n; j++) {
        if (dp[fullMask][j] < bestCost) {
            bestCost = dp[fullMask][j];
            bestEnd = j;
        }
    }

    // Reconstruct the path by walking the parent pointers backward from the end
    const order = [];
    let mask = fullMask;
    let curr = bestEnd;
    while (curr !== -1) {
        order.push(curr);
        const prev = parent[mask][curr];
        mask ^= (1 << curr);
        curr = prev;
    }
    order.reverse();
    return order;
};

