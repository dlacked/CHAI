const darkCheckbox = document.getElementById('dark-mode-checkbox');
const fileInput = document.getElementById('oral-image-input');
const fileStatus = document.getElementById('oral-image-status');

// 2D VIEW Checkboxes
const jawCb = document.getElementById('jaw-classification-checkbox');
const yoloCb = document.getElementById('yolov8-seg-checkbox');
const pcaCb = document.getElementById('pca-checkbox');
const archPathCb = document.getElementById('arch-path-checkbox');
const gapCb = document.getElementById('gap-checkbox');
const fdiCb = document.getElementById('fdi-checkbox');


let imageFiles = [];
let currentImageIndex = 0;
let currentObjectURL = null;
let currentImage = null;
const classificationCache = {};
const segmentationCache = {};

const canvas = document.getElementById('raw-canvas');
const ctx = canvas.getContext('2d');
const currentSliceDisplay = document.getElementById('current-slice');
const fileNameDisplay = document.getElementById('file-name');

const applyMode = (dark) => {
    document.body.classList.toggle('light-mode', !dark);
    localStorage.setItem('darkMode', dark ? '1' : '0');
};

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

    return { normWeight, ptA: res.ptA, ptB: res.ptB, distance: res.distance };
};

// Walks the arch order and splits it into groups wherever two consecutive teeth are not
// touching (normalized gap > TOUCH_NORM_WEIGHT). A missing tooth breaks the chain, so a
// single image can produce more than one group - the break between groups is the gap left
// by the missing tooth.
const computeArchGroups = (predictions, order, W_pca, H_pca, pca) => {
    if (order.length === 0) return [];

    const groups = [[order[0]]];
    for (let i = 1; i < order.length; i++) {
        const gap = computeGapNormWeight(predictions, order[i - 1], order[i], W_pca, H_pca, pca);
        if (gap && gap.normWeight <= TOUCH_NORM_WEIGHT) {
            groups[groups.length - 1].push(order[i]);
        } else {
            groups.push([order[i]]);
        }
    }
    return groups;
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

// Draws a black badge with a '#' + FDI number (white '#', cyan number) centered at the given point
const drawFdiNumberBadge = (centerX, centerY, numText) => {
    const boxWidth = 76;
    const boxHeight = 50;
    const rx = centerX - boxWidth / 2;
    const ry = centerY - boxHeight / 2;

    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') {
        ctx.roundRect(rx, ry, boxWidth, boxHeight, 8);
    } else {
        ctx.rect(rx, ry, boxWidth, boxHeight);
    }
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.8;
    ctx.stroke();

    const hashText = '#';

    ctx.font = 'bold 30px sans-serif';
    const numWidth = ctx.measureText(numText).width;
    ctx.font = 'bold 16px sans-serif';
    const hashWidth = ctx.measureText(hashText).width;

    const spacing = 2;
    const totalWidth = hashWidth + spacing + numWidth;
    const startX = centerX - totalWidth / 2;

    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';

    // Draw '#' in small font (white)
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px sans-serif';
    ctx.fillText(hashText, startX, centerY + 1);

    // Draw the number in large font (cyan)
    ctx.fillStyle = '#00ffff';
    ctx.font = 'bold 30px sans-serif';
    ctx.fillText(numText, startX + hashWidth + spacing, centerY + 1);
};

const redrawCanvas = () => {
    if (!currentImage) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const file = imageFiles[currentImageIndex];
    const isUpper = file && classificationCache[file.name] && classificationCache[file.name].class === 'upper';
    const isLower = file && classificationCache[file.name] && classificationCache[file.name].class === 'lower';
    const hasClassification = isUpper || isLower;
    const predictions = file ? segmentationCache[file.name] : null;

    let pca = null;
    if (hasClassification && predictions) {
        pca = calculatePCARotation(predictions, canvas.width, isUpper);
    }

    ctx.save();
    if (pca && pcaCb.checked && !pcaCb.disabled) {
        // Rotate around the PCA center
        ctx.translate(pca.center.x, pca.center.y);
        ctx.rotate(-pca.angle);
        ctx.translate(-pca.center.x, -pca.center.y);
    }

    // Draw the main image
    ctx.drawImage(currentImage, 0, 0);

    // Draw YOLO Segmentation if checked
    if (yoloCb.checked && !yoloCb.disabled && predictions) {
        drawSegmentation(predictions);
    }

    // Draw PCA overlay if PCA is active
    if (pca && pcaCb.checked && !pcaCb.disabled) {
        // Draw the connection line (horizontal reference in rotated context)
        ctx.beginPath();
        ctx.moveTo(pca.pLeft.x, pca.pLeft.y);
        ctx.lineTo(pca.pRight.x, pca.pRight.y);
        ctx.strokeStyle = '#ff3366';
        ctx.lineWidth = 3;
        ctx.setLineDash([6, 6]);
        ctx.stroke();
        ctx.setLineDash([]); // Reset

        // Draw left topmost point
        ctx.beginPath();
        ctx.arc(pca.pLeft.x, pca.pLeft.y, 8, 0, 2 * Math.PI);
        ctx.fillStyle = '#ff3366';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw right topmost point
        ctx.beginPath();
        ctx.arc(pca.pRight.x, pca.pRight.y, 8, 0, 2 * Math.PI);
        ctx.fillStyle = '#ff3366';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw center pivot point
        ctx.beginPath();
        ctx.arc(pca.center.x, pca.center.y, 10, 0, 2 * Math.PI);
        ctx.fillStyle = '#00e5ff';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2.5;
        ctx.stroke();
    }

    // Draw Arch Path if checked
    if (archPathCb.checked && !archPathCb.disabled && predictions) {
        const archPathStatus = document.getElementById('arch-path-status');
        if (predictions.length < 2) {
            if (archPathStatus) {
                archPathStatus.textContent = ' Requires 2+';
                archPathStatus.style.color = '#f44336';
            }
        } else {
            if (archPathStatus) {
                archPathStatus.textContent = ' Active';
                archPathStatus.style.color = '#4caf50';
            }

            // Compute centroids
            const vertices = predictions.map(pred => computeCentroid(pred.polygon));

            // Nearest-neighbor greedy order through all teeth
            const order = computeArchOrder(vertices, pca, isUpper);

            // Draw path edges (dashed lines) - no distance labels for now
            for (let i = 0; i < order.length - 1; i++) {
                const uPt = vertices[order[i]];
                const vPt = vertices[order[i + 1]];

                ctx.beginPath();
                ctx.moveTo(uPt.x, uPt.y);
                ctx.lineTo(vPt.x, vPt.y);
                ctx.strokeStyle = '#00ff66'; // Vibrant green
                ctx.lineWidth = 2.5;
                ctx.setLineDash([5, 5]);
                ctx.stroke();
                ctx.setLineDash([]); // Reset
            }

            // Draw vertices (centroids)
            vertices.forEach(pt => {
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 6, 0, 2 * Math.PI);
                ctx.fillStyle = '#00ff66';
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            });

        }
    } else {
        const archPathStatus = document.getElementById('arch-path-status');
        if (archPathStatus && !archPathCb.disabled) archPathStatus.textContent = '';
    }

    // coords/theta/seq rendering removed from here, integrated into FDI drawing block below

    // Draw Segment Gap if checked - labels the distance for teeth that are NOT touching
    if (gapCb.checked && !gapCb.disabled && predictions) {
        const gapStatus = document.getElementById('gap-status');
        if (predictions.length < 2) {
            if (gapStatus) {
                gapStatus.textContent = ' Requires 2+';
                gapStatus.style.color = '#f44336';
            }
        } else if (predictions.length >= 12) {
            if (gapStatus) {
                gapStatus.textContent = ' Deactivated';
                gapStatus.style.color = '#f44336';
            }
        } else {
            if (gapStatus) {
                gapStatus.textContent = ' Active';
                gapStatus.style.color = '#4caf50';
            }

            // Walk the tooth arch in nearest-neighbor order and compare each tooth's segment
            // only to the next one in that order - this follows the arch's actual curve
            // (unlike a plain X-sort, which breaks down where the arch bends back on itself)
            const vertices = predictions.map(pred => computeCentroid(pred.polygon));
            const order = computeArchOrder(vertices, pca, isUpper);
            const { W_pca, H_pca } = computeArchPcaBounds(predictions, pca, canvas.width, canvas.height);

            // Needed to label each gap with the FDI number of the tooth that's missing there
            const slots = hasClassification ? computeArchSlots(predictions, order, W_pca, H_pca, pca) : null;
            const fdiLabels = isLower
                ? [46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36]
                : [16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26];

            for (let i = 0; i < order.length - 1; i++) {
                const gap = computeGapNormWeight(predictions, order[i], order[i + 1], W_pca, H_pca, pca);
                if (!gap) continue;
                if (gap.normWeight <= TOUCH_NORM_WEIGHT) continue; // touching - not a gap

                const { ptA, ptB } = gap;

                // Draw gap edge (dashed line) between the two closest polygon points
                ctx.beginPath();
                ctx.moveTo(ptA[0], ptA[1]);
                ctx.lineTo(ptB[0], ptB[1]);
                ctx.strokeStyle = '#ff3366'; // Neon pink - distinguishes gaps from the Arch Path's green
                ctx.lineWidth = 2.5;
                ctx.setLineDash([5, 5]);
                ctx.stroke();
                ctx.setLineDash([]); // Reset

                // The missing tooth's FDI number sits in the slot right after the tooth before the gap
                const missingSlot = slots ? slots[order[i]] + 1 : -1;
                const missingLabel = missingSlot >= 0 && missingSlot < fdiLabels.length
                    ? String(fdiLabels[missingSlot])
                    : '?';

                // Draw a three-line label at the middle of the edge: missing FDI number, "LOSS", and then the distance
                const midX = (ptA[0] + ptB[0]) / 2;
                const midY = (ptA[1] + ptB[1]) / 2;
                const valText = missingLabel;
                const lossText = 'LOSS';
                const distText = `${gap.normWeight.toFixed(4)}`;

                ctx.font = 'bold 28px sans-serif';
                const valWidth = ctx.measureText(valText).width;
                const valHeight = 28;

                ctx.font = 'bold 20px sans-serif';
                const lossWidth = ctx.measureText(lossText).width;
                const lossHeight = 20;

                ctx.font = '14px sans-serif';
                const distWidth = ctx.measureText(distText).width;
                const distHeight = 14;

                const padding = 12;
                const lineGap = 4;
                const boxWidth = Math.max(valWidth, lossWidth, distWidth) + padding * 2;
                const boxHeight = padding * 2 + valHeight + lineGap + lossHeight + lineGap + distHeight;
                const boxX = midX - boxWidth / 2;
                const boxY = midY - boxHeight / 2;
                const valY = boxY + padding;
                const lossY = valY + valHeight + lineGap;
                const distY = lossY + lossHeight + lineGap;

                // Text background box for legibility
                ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
                ctx.beginPath();
                if (typeof ctx.roundRect === 'function') {
                    ctx.roundRect(boxX, boxY, boxWidth, boxHeight, 6);
                } else {
                    ctx.rect(boxX, boxY, boxWidth, boxHeight);
                }
                ctx.fill();
                ctx.strokeStyle = '#ff3366';
                ctx.lineWidth = 1;
                ctx.stroke();

                // Missing FDI number (top line, red)
                ctx.font = 'bold 28px sans-serif';
                ctx.fillStyle = '#ff3366';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(valText, midX, valY);

                // "LOSS" label (middle line, white)
                ctx.font = 'bold 20px sans-serif';
                ctx.fillStyle = '#ffffff';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(lossText, midX, lossY);

                // Distance label (bottom line, soft pink)
                ctx.font = '14px sans-serif';
                ctx.fillStyle = '#ffb3c6';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(distText, midX, distY);
            }
        }
    } else {
        const gapStatus = document.getElementById('gap-status');
        if (gapStatus && !gapCb.disabled) gapStatus.textContent = '';
    }

    // Draw FDI numbers if checked - labels teeth in nearest-neighbor arch order regardless of tooth count
    if (fdiCb.checked && !fdiCb.disabled && predictions && pca) {
        const fdiStatus = document.getElementById('fdi-status');

        if (!hasClassification) {
            if (fdiStatus) {
                fdiStatus.textContent = ' Requires Jaw Classification';
                fdiStatus.style.color = '#f44336';
            }
        } else {
            if (fdiStatus) {
                fdiStatus.textContent = ' Active';
                fdiStatus.style.color = '#4caf50';
            }

            const vertices = predictions.map(pred => computeCentroid(pred.polygon));
            const order = computeArchOrder(vertices, pca, isUpper);
            const { W_pca, H_pca } = computeArchPcaBounds(predictions, pca, canvas.width, canvas.height);
            const slots = computeArchSlots(predictions, order, W_pca, H_pca, pca);

            const fdiLabels = isLower
                ? [46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36]
                : [16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26];

            const drawFdiLabels = () => {
                order.forEach(vertexIdx => {
                    const slot = slots[vertexIdx];
                    if (slot < 0 || slot >= fdiLabels.length) return;

                    const pred = predictions[vertexIdx];
                    const centerX = (pred.box[0] + pred.box[2]) / 2;
                    const centerY = (pred.box[1] + pred.box[3]) / 2;
                    drawFdiNumberBadge(centerX, centerY, String(fdiLabels[slot]));
                });
            };

            if (predictions.length === 12) {
                drawFdiLabels();
            } else {
                // Fewer than 12 teeth detected. Group the arch order into touching-chains
                // separated by gaps (each gap = exactly one missing tooth by assumption). If
                // detected teeth + gaps reconstructs to exactly 12, the slot assignment is
                // trustworthy - label normally. Otherwise fall back to the raw debug view.
                const groups = computeArchGroups(predictions, order, W_pca, H_pca, pca);
                const reconstructedTotal = predictions.length + (groups.length - 1);

                if (reconstructedTotal === 12) {
                    drawFdiLabels();
                } else {
                    // 11 or fewer teeth (and the gap count doesn't reconstruct to 12): draw
                    // coordinates, theta, and greedy sequence numbers instead
                    predictions.forEach((pred, vertexIdx) => {
                        const poly = pred.polygon;
                        if (!poly || poly.length === 0) return;
    
                        // 1. Calculate bounding box of polygon (raw screen coordinates)
                        const xs = poly.map(pt => pt[0]);
                        const ys = poly.map(pt => pt[1]);
                        const x_min = Math.min(...xs);
                        const x_max = Math.max(...xs);
                        const y_min = Math.min(...ys);
                        const y_max = Math.max(...ys);
    
                        // 2. Draw bounding box
                        ctx.beginPath();
                        ctx.rect(x_min, y_min, x_max - x_min, y_max - y_min);
                        ctx.strokeStyle = '#f39c12'; // Orange-yellow bounding box
                        ctx.lineWidth = 2;
                        ctx.stroke();
    
                        // 3. Compute normalized features
                        const centroid = computeCentroid(poly);
                        const rx = centroid.x - pca.center.x;
                        const ry = centroid.y - pca.center.y;
                        const cos = Math.cos(-pca.angle);
                        const sin = Math.sin(-pca.angle);
                        const tx = rx * cos - ry * sin;
                        const ty = rx * sin + ry * cos;
                        const angle_rad = Math.atan2(ty, tx);
                        const angle_norm = angle_rad / Math.PI;
    
                        const slot = slots ? slots[vertexIdx] : -1;
                        const fdi_number = (slot >= 0 && slot < fdiLabels.length) ? fdiLabels[slot] : -1;
    
                        // Center, mirror (x flip), and normalize coordinates
                        let x1_c = x_min - pca.center.x;
                        let y1_c = y_min - pca.center.y;
                        let x2_c = x_max - pca.center.x;
                        let y2_c = y_max - pca.center.y;
    
                        if (fdi_number !== -1) {
                            const tens = Math.floor(fdi_number / 10);
                            if (tens === 2 || tens === 3) {
                                const x1_new = -x1_c;
                                const x2_new = -x2_c;
                                x1_c = Math.min(x1_new, x2_new);
                                x2_c = Math.max(x1_new, x2_new);
                            }
                        }
    
                        const x1_norm = x1_c / canvas.width;
                        const y1_norm = y1_c / canvas.height;
                        const x2_norm = x2_c / canvas.width;
                        const y2_norm = y2_c / canvas.height;
    
                        // Mirror theta
                        let theta_val = angle_norm;
                        if (fdi_number !== -1) {
                            const tens = Math.floor(fdi_number / 10);
                            if (tens === 2 || tens === 3) {
                                if (theta_val >= 0) theta_val = 1.0 - theta_val;
                                else theta_val = -1.0 - theta_val;
                            }
                        }
    
                        const mst_seq = order.indexOf(vertexIdx);
    
                        // 4. Build text lines
                        const lines = [];
                        lines.push(`Seq: ${mst_seq}`);
                        lines.push(`x1:${x1_norm.toFixed(4)} y1:${y1_norm.toFixed(4)}`);
                        lines.push(`x2:${x2_norm.toFixed(4)} y2:${y2_norm.toFixed(4)}`);
                        lines.push(`θ:${theta_val.toFixed(6)}`);
    
                        // 5. Render label box at centroid
                        ctx.font = 'bold 20px monospace';
                        const lineHeight = 24;
                        const boxPadding = 12;
    
                        let maxTextWidth = 0;
                        lines.forEach(line => {
                            const w = ctx.measureText(line).width;
                            if (w > maxTextWidth) maxTextWidth = w;
                        });
    
                        const labelBoxWidth = maxTextWidth + boxPadding * 2;
                        const labelBoxHeight = lines.length * lineHeight + boxPadding * 2;
    
                        const labelBoxX = centroid.x - labelBoxWidth / 2;
                        const labelBoxY = centroid.y - labelBoxHeight / 2;
    
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
                        ctx.beginPath();
                        if (typeof ctx.roundRect === 'function') {
                            ctx.roundRect(labelBoxX, labelBoxY, labelBoxWidth, labelBoxHeight, 4);
                        } else {
                            ctx.rect(labelBoxX, labelBoxY, labelBoxWidth, labelBoxHeight);
                        }
                        ctx.fill();
                        ctx.strokeStyle = '#f39c12';
                        ctx.lineWidth = 1;
                        ctx.stroke();
    
                        ctx.textAlign = 'left';
                        ctx.textBaseline = 'top';
    
                        lines.forEach((line, i) => {
                            const lineY = labelBoxY + boxPadding + i * lineHeight;
                            if (line.startsWith('Seq:')) {
                                ctx.fillStyle = '#00ff66';
                            } else if (line.startsWith('θ')) {
                                ctx.fillStyle = '#ff80df';
                            } else {
                                ctx.fillStyle = '#ffffff';
                            }
                            ctx.fillText(line, labelBoxX + boxPadding, lineY);
                        });
                    });
                }
            }
        }
    } else {
        const fdiStatus = document.getElementById('fdi-status');
        if (fdiStatus && !fdiCb.disabled) fdiStatus.textContent = '';
    }

    ctx.restore();
};

const drawSegmentation = (predictions) => {
    if (!predictions || predictions.length === 0) return;

    predictions.forEach((pred, idx) => {
        const poly = pred.polygon;
        if (!poly || poly.length === 0) return;

        ctx.beginPath();
        ctx.moveTo(poly[0][0], poly[0][1]);
        for (let i = 1; i < poly.length; i++) {
            ctx.lineTo(poly[i][0], poly[i][1]);
        }
        ctx.closePath();

        // Use golden ratio color distribution for distinct, vibrant colors
        const hue = (idx * 137.5) % 360;
        ctx.fillStyle = `hsla(${hue}, 70%, 50%, 0.45)`;
        ctx.fill();

        ctx.strokeStyle = `hsla(${hue}, 70%, 50%, 0.95)`;
        ctx.lineWidth = 3;
        ctx.stroke();
    });
};

const displayImage = (index) => {
    if (index < 0 || index >= imageFiles.length) return;
    const file = imageFiles[index];

    if (currentObjectURL) {
        URL.revokeObjectURL(currentObjectURL);
    }

    currentObjectURL = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        currentImage = img;

        redrawCanvas();

        currentSliceDisplay.textContent = `Image: ${index + 1}/${imageFiles.length}`;
        fileNameDisplay.textContent = `File: ${file.name}`;
        fileStatus.textContent = `Loaded ${imageFiles.length} images.`;
    };
    img.src = currentObjectURL;

    // 이미지 바뀔 때마다 classification 및 segmentation 새로 수행
    if (!jawCb.disabled && jawCb.checked) {
        classifyJawImage(file);
    } else {
        document.getElementById('jaw-classification-result').textContent = '';
    }

    if (!yoloCb.disabled && yoloCb.checked) {
        segmentImage(file);
    } else {
        const resultSpan = document.getElementById('yolo-segmentation-result');
        if (resultSpan) resultSpan.textContent = '';
    }
};

const classifyJawImage = (file) => {
    const resultSpan = document.getElementById('jaw-classification-result');

    if (classificationCache[file.name]) {
        const data = classificationCache[file.name];
        const isUpper = data.class === 'upper';
        resultSpan.textContent = isUpper ? ' Maxilla (Upper Jaw)' : ' Mandible (Lower Jaw)';
        resultSpan.style.color = '#4caf50';
        jawCb.checked = true;
        updateCheckboxStates();
        return;
    }

    resultSpan.textContent = ' Classifying...';
    resultSpan.style.color = 'rgba(255, 255, 255, 0.5)';

    const formData = new FormData();
    formData.append('image', file);

    fetch('/classify', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                classificationCache[file.name] = data;
                const isUpper = data.class === 'upper';
                resultSpan.textContent = isUpper ? ' Maxilla (Upper Jaw)' : ' Mandible (Lower Jaw)';
                resultSpan.style.color = '#4caf50';

                jawCb.checked = true;
                updateCheckboxStates();
            } else {
                resultSpan.textContent = ' Error';
                resultSpan.style.color = '#f44336';
                console.error('Classification error:', data.error);
            }
        })
        .catch(error => {
            resultSpan.textContent = ' Offline';
            resultSpan.style.color = '#f44336';
            console.error('Server offline or network error:', error);
        });
};

const segmentImage = (file) => {
    const resultSpan = document.getElementById('yolo-segmentation-result');

    if (segmentationCache[file.name]) {
        if (resultSpan) {
            resultSpan.textContent = ` Done (${segmentationCache[file.name].length})`;
            resultSpan.style.color = '#4caf50';
        }
        redrawCanvas();
        return;
    }

    if (resultSpan) {
        resultSpan.textContent = ' Segmenting...';
        resultSpan.style.color = 'rgba(255, 255, 255, 0.5)';
    }

    const formData = new FormData();
    formData.append('image', file);

    fetch('/segment', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                segmentationCache[file.name] = data.predictions;
                if (resultSpan) {
                    resultSpan.textContent = ` Done (${data.predictions.length})`;
                    resultSpan.style.color = '#4caf50';
                }
                redrawCanvas();
            } else {
                if (resultSpan) {
                    resultSpan.textContent = ' Error';
                    resultSpan.style.color = '#f44336';
                }
                console.error('Segmentation error:', data.error);
            }
        })
        .catch(error => {
            if (resultSpan) {
                resultSpan.textContent = ' Offline';
                resultSpan.style.color = '#f44336';
            }
            console.error('Server offline or network error:', error);
        });
};

fileInput.addEventListener('change', () => {
    const files = Array.from(fileInput.files || []);
    if (files.length === 0) {
        fileStatus.textContent = 'No directory selected';
        jawCb.disabled = true;
        jawCb.checked = false;
        updateCheckboxStates();
        document.getElementById('jaw-classification-result').textContent = '';
        const resultSpan = document.getElementById('yolo-segmentation-result');
        if (resultSpan) resultSpan.textContent = '';
        return;
    }

    imageFiles = files
        .filter(file => {
            const ext = file.name.toLowerCase().split('.').pop();
            return ['jpg', 'jpeg', 'png'].includes(ext) || file.type.startsWith('image/');
        })
        .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }));

    if (imageFiles.length === 0) {
        fileStatus.textContent = 'No JPG, JPEG, or PNG images found in directory.';
        fileInput.value = '';
        currentSliceDisplay.textContent = 'Image: 0/0';
        fileNameDisplay.textContent = 'File: Not Opened';
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        jawCb.disabled = true;
        jawCb.checked = false;
        updateCheckboxStates();
        document.getElementById('jaw-classification-result').textContent = '';
        const resultSpan = document.getElementById('yolo-segmentation-result');
        if (resultSpan) resultSpan.textContent = '';
        return;
    }

    currentImageIndex = 0;
    displayImage(currentImageIndex);

    // Enable checkbox and trigger classification
    jawCb.disabled = false;
    classifyJawImage(imageFiles[0]);
});

const container = document.getElementById('2d-container');
container.addEventListener('wheel', (event) => {
    if (imageFiles.length === 0) return;

    event.preventDefault();

    if (event.deltaY > 0) {
        // 마지막 이미지에서 휠을 내리면 처음 이미지로 이동
        if (currentImageIndex < imageFiles.length - 1) {
            currentImageIndex++;
        } else {
            currentImageIndex = 0;
        }
        displayImage(currentImageIndex);
    } else if (event.deltaY < 0) {
        // 첫 번째 이미지에서 휠을 올리면 마지막 이미지로 이동
        if (currentImageIndex > 0) {
            currentImageIndex--;
        } else {
            currentImageIndex = imageFiles.length - 1;
        }
        displayImage(currentImageIndex);
    }
}, { passive: false });

darkCheckbox.addEventListener('change', () => applyMode(darkCheckbox.checked));
const saved = localStorage.getItem('darkMode');
if (saved === '0') {
    darkCheckbox.checked = false;
    applyMode(false);
}

// 2D VIEW Checkbox Dependency Logic
const updateCheckboxStates = () => {
    yoloCb.disabled = !jawCb.checked;
    if (yoloCb.disabled) {
        yoloCb.checked = false;
        const resultSpan = document.getElementById('yolo-segmentation-result');
        if (resultSpan) resultSpan.textContent = '';
    }

    pcaCb.disabled = !yoloCb.checked;
    if (pcaCb.disabled) pcaCb.checked = false;

    // Arch Path is enabled when PCA is checked
    archPathCb.disabled = !pcaCb.checked;
    if (archPathCb.disabled) {
        archPathCb.checked = false;
        const archPathStatus = document.getElementById('arch-path-status');
        if (archPathStatus) archPathStatus.textContent = '';
    }

    const file = imageFiles[currentImageIndex];
    const predictions = file ? segmentationCache[file.name] : null;
    const is12Teeth = predictions && predictions.length === 12;

    // Segment Gap is enabled when Arch Path is checked. With exactly 12 teeth detected there's
    // nothing to show (no possible gaps), but the checkbox itself stays checkable regardless.
    gapCb.disabled = !archPathCb.checked;
    if (gapCb.disabled) {
        gapCb.checked = false;
        const gapStatus = document.getElementById('gap-status');
        if (gapStatus) gapStatus.textContent = '';
    } else {
        const gapStatus = document.getElementById('gap-status');
        if (gapStatus) {
            if (gapCb.checked) {
                gapStatus.textContent = ' Active';
                gapStatus.style.color = '#4caf50';
            } else if (is12Teeth) {
                gapStatus.textContent = ' Deactivated';
                gapStatus.style.color = '#f44336';
            } else {
                gapStatus.textContent = '';
            }
        }
    }

    // FDI is enabled when Arch Path is checked
    fdiCb.disabled = !archPathCb.checked;
    if (fdiCb.disabled) {
        fdiCb.checked = false;
        const fdiStatus = document.getElementById('fdi-status');
        if (fdiStatus) fdiStatus.textContent = '';
    } else {
        const fdiStatus = document.getElementById('fdi-status');
        if (fdiStatus) {
            fdiStatus.textContent = fdiCb.checked ? ' Active' : '';
            fdiStatus.style.color = fdiCb.checked ? '#4caf50' : '';
        }
    }
};

jawCb.addEventListener('change', () => {
    updateCheckboxStates();
    if (jawCb.checked && imageFiles.length > 0) {
        classifyJawImage(imageFiles[currentImageIndex]);
    } else if (!jawCb.checked) {
        document.getElementById('jaw-classification-result').textContent = '';
    }
});

yoloCb.addEventListener('change', () => {
    updateCheckboxStates();
    const resultSpan = document.getElementById('yolo-segmentation-result');
    if (yoloCb.checked) {
        if (imageFiles.length > 0) {
            segmentImage(imageFiles[currentImageIndex]);
        }
    } else {
        if (resultSpan) resultSpan.textContent = '';
        redrawCanvas();
    }
});

pcaCb.addEventListener('change', () => {
    updateCheckboxStates();
    redrawCanvas();
});

archPathCb.addEventListener('change', () => {
    updateCheckboxStates();
    redrawCanvas();
});



gapCb.addEventListener('change', () => {
    updateCheckboxStates();
    redrawCanvas();
});

fdiCb.addEventListener('change', () => {
    updateCheckboxStates();
    redrawCanvas();
});

// Initialize states
jawCb.disabled = true;
updateCheckboxStates();
