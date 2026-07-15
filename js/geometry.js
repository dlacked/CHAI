const calculatePCARotation = (predictions, canvasWidth, isUpper) => {
    if (!predictions || predictions.length === 0) return null;

    let leftPoints = [];
    let rightPoints = [];

    predictions.forEach(pred => {
        const poly = pred.polygon;
        if (!poly || poly.length === 0) return;

        poly.forEach(pt => {
            const x = pt[0];
            const y = pt[1];

            if (x < canvasWidth / 2) {
                leftPoints.push({ x, y });
            } else {
                rightPoints.push({ x, y });
            }
        });
    });

    if (leftPoints.length === 0 || rightPoints.length === 0) return null;

    // targetY 찾기 (lower의 경우 min y, upper의 경우 max y)
    let pLeft = null;
    leftPoints.forEach(pt => {
        if (!pLeft) {
            pLeft = pt;
            return;
        }

        const isTarget = isUpper ? (pt.y > pLeft.y) : (pt.y < pLeft.y);
        const isTie = Math.abs(pt.y - pLeft.y) < 1e-4;

        if (isTarget) {
            pLeft = pt;
        } else if (isTie) {
            // 좌측 영역 tie-breaker: 가장 작은 x (leftmost)
            if (pt.x < pLeft.x) {
                pLeft = pt;
            }
        }
    });

    let pRight = null;
    rightPoints.forEach(pt => {
        if (!pRight) {
            pRight = pt;
            return;
        }

        const isTarget = isUpper ? (pt.y > pRight.y) : (pt.y < pRight.y);
        const isTie = Math.abs(pt.y - pRight.y) < 1e-4;

        if (isTarget) {
            pRight = pt;
        } else if (isTie) {
            // 우측 영역 tie-breaker: 가장 큰 x (rightmost)
            if (pt.x > pRight.x) {
                pRight = pt;
            }
        }
    });

    if (!pLeft || !pRight) return null;

    const centerX = (pLeft.x + pRight.x) / 2;
    const centerY = (pLeft.y + pRight.y) / 2;
    const angle = Math.atan2(pRight.y - pLeft.y, pRight.x - pLeft.x);

    return {
        pLeft,
        pRight,
        center: { x: centerX, y: centerY },
        angle: angle
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

// Builds a path through all teeth via nearest-neighbor greedy: starting from the leftmost
// tooth, repeatedly jump to the closest unvisited tooth. Unlike a general MST, this always
// produces a simple chain - no tooth can end up with more than two neighbors.
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

    const visited = new Array(n).fill(false);
    visited[startIdx] = true;
    const order = [startIdx];

    while (order.length < n) {
        const curr = vertices[order[order.length - 1]];
        let nextIdx = -1;
        let minDist = Infinity;
        for (let i = 0; i < n; i++) {
            if (visited[i]) continue;
            const dx = vertices[i].x - curr.x;
            const dy = vertices[i].y - curr.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < minDist) {
                minDist = dist;
                nextIdx = i;
            }
        }
        if (nextIdx === -1) break;
        visited[nextIdx] = true;
        order.push(nextIdx);
    }
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
