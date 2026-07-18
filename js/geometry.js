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

// The FDI quadrant (tens digit) only depends on jaw (upper/lower) + which side of the PCA
// centerline a tooth sits on - unlike slot-based lookup, this works even when the tooth count
// doesn't reconstruct to 12, so it's used for the Holding-state FDI badges (ResNet fallback)
const computeQuadrantTens = (rotatedX, isUpper) => {
    const isLeftSide = rotatedX < 0;
    if (isUpper) return isLeftSide ? 1 : 2;
    return isLeftSide ? 4 : 3;
};

// Computes the same normalized [x1, y1, x2, y2, theta] independent variables the ResNet Tooth
// model was trained on (see functions/features/coords.py + theta.py) from a live prediction's
// box + polygon, using the shared PCA from calculatePCARotation(). Mirroring is decided purely
// by which side of the PCA centerline the tooth's centroid falls on (positive rotated-x = the
// mirrored FDI quadrant), matching the rule used to build the training CSVs - so this works
// without knowing the tooth's FDI number up front.
const computeToothMeta = (pred, pca) => {
    const centroid = computeCentroid(pred.polygon);
    const { tx, ty } = rotateToPcaFrame(centroid.x, centroid.y, pca);
    const mirror = tx >= 0;

    let x1_c = pred.box[0] - pca.center.x;
    let y1_c = pred.box[1] - pca.center.y;
    let x2_c = pred.box[2] - pca.center.x;
    let y2_c = pred.box[3] - pca.center.y;

    if (mirror) {
        const x1_new = -x1_c;
        const x2_new = -x2_c;
        x1_c = Math.min(x1_new, x2_new);
        x2_c = Math.max(x1_new, x2_new);
    }

    let theta = Math.atan2(ty, tx) / Math.PI;
    if (mirror) {
        theta = theta >= 0 ? 1.0 - theta : -1.0 - theta;
    }

    return {
        x1: x1_c / canvas.width,
        y1: y1_c / canvas.height,
        x2: x2_c / canvas.width,
        y2: y2_c / canvas.height,
        theta
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

// Two adjacent teeth whose normalized gap is at or below this are considered touching
const TOUCH_NORM_WEIGHT = 0.05;

// Finds the closest pair of points between two tooth polygons (the actual segment-to-segment gap)
const getClosestPolygonPoints = (polyA, polyB) => {
    let minDist = Infinity;
    let ptA = null;
    let ptB = null;
    if (!polyA || !polyB || polyA.length === 0 || polyB.length === 0) {
        return { distance: minDist, ptA, ptB };
    }
    for (let i = 0; i < polyA.length; i++) {
        const pA = polyA[i];
        for (let j = 0; j < polyB.length; j++) {
            const pB = polyB[j];
            const dx = pA[0] - pB[0];
            const dy = pA[1] - pB[1];
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < minDist) {
                minDist = dist;
                ptA = pA;
                ptB = pB;
            }
        }
    }
    return { distance: minDist, ptA, ptB };
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

// The tooth arch's bounding size in the PCA-rotated coordinate system, used to normalize
// pixel distances into a 0..1 scale that's comparable across images of different sizes/zooms
const computeArchPcaBounds = (predictions, pca, canvasWidth, canvasHeight) => {
    let W_pca = canvasWidth;
    let H_pca = canvasHeight;
    if (pca) {
        const cos = Math.cos(-pca.angle);
        const sin = Math.sin(-pca.angle);

        let minTx = Infinity, maxTx = -Infinity;
        let minTy = Infinity, maxTy = -Infinity;

        predictions.forEach(pred => {
            if (!pred.polygon) return;
            pred.polygon.forEach(pt => {
                const rx = pt[0] - pca.center.x;
                const ry = pt[1] - pca.center.y;
                const tx = rx * cos - ry * sin;
                const ty = rx * sin + ry * cos;
                if (tx < minTx) minTx = tx;
                if (tx > maxTx) maxTx = tx;
                if (ty < minTy) minTy = ty;
                if (ty > maxTy) maxTy = ty;
            });
        });

        if (maxTx > minTx) W_pca = maxTx - minTx;
        if (maxTy > minTy) H_pca = maxTy - minTy;
    }
    return { W_pca, H_pca };
};

// The actual segment-to-segment gap between two teeth, normalized against the arch size.
// Returns null if either polygon is missing.
const computeGapNormWeight = (predictions, idxA, idxB, W_pca, H_pca, pca) => {
    const res = getClosestPolygonPoints(predictions[idxA].polygon, predictions[idxB].polygon);
    if (!res.ptA || !res.ptB) return null;

    let dxRot = res.ptB[0] - res.ptA[0];
    let dyRot = res.ptB[1] - res.ptA[1];

    if (pca) {
        const cos = Math.cos(-pca.angle);
        const sin = Math.sin(-pca.angle);

        const rxU = res.ptA[0] - pca.center.x;
        const ryU = res.ptA[1] - pca.center.y;
        const txU = rxU * cos - ryU * sin;
        const tyU = rxU * sin + ryU * cos;

        const rxV = res.ptB[0] - pca.center.x;
        const ryV = res.ptB[1] - pca.center.y;
        const txV = rxV * cos - ryV * sin;
        const tyV = rxV * sin + ryV * cos;

        dxRot = txV - txU;
        dyRot = tyV - tyU;
    }

    const normDx = W_pca > 0 ? dxRot / W_pca : 0;
    const normDy = H_pca > 0 ? dyRot / H_pca : 0;
    const normWeight = Math.min(1.0, Math.sqrt(normDx * normDx + normDy * normDy));

    return { normWeight, ptA: res.ptA, ptB: res.ptB };
};

// Assigns each tooth in the arch order an FDI slot (0-11), leaving the slot gap left by a
// missing tooth wherever two consecutive teeth are not touching. Returns an array indexed by
// the original prediction index (values are -1 for teeth that don't fit within 12 slots).
const computeArchSlots = (predictions, order, W_pca, H_pca, pca) => {
    const slots = new Array(predictions.length).fill(-1);
    if (predictions.length === 12) {
        order.forEach((idx, i) => {
            slots[idx] = i;
        });
        return slots;
    }
    let slot = 0;
    order.forEach((idx, i) => {
        if (i > 0) {
            const gap = computeGapNormWeight(predictions, order[i - 1], idx, W_pca, H_pca, pca);
            if (!gap || gap.normWeight > TOUCH_NORM_WEIGHT) slot += 1; // skip the missing tooth's slot
        }
        slots[idx] = slot;
        slot += 1;
    });
    return slots;
};
