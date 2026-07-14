const darkCheckbox = document.getElementById('dark-mode-checkbox');
const fileInput = document.getElementById('oral-image-input');
const fileStatus = document.getElementById('oral-image-status');

// 2D VIEW Checkboxes
const jawCb = document.getElementById('jaw-classification-checkbox');
const yoloCb = document.getElementById('yolov8-seg-checkbox');
const pcaCb = document.getElementById('pca-checkbox');
const mstCb = document.getElementById('mst-checkbox');
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

const computeMST = (vertices) => {
    const n = vertices.length;
    if (n === 0) return [];

    const inMST = new Array(n).fill(false);
    const minEdge = new Array(n).fill(Infinity);
    const parent = new Array(n).fill(-1);

    minEdge[0] = 0;
    const edges = [];

    for (let count = 0; count < n; count++) {
        let u = -1;
        let minVal = Infinity;
        for (let i = 0; i < n; i++) {
            if (!inMST[i] && minEdge[i] < minVal) {
                minVal = minEdge[i];
                u = i;
            }
        }

        if (u === -1) break;

        inMST[u] = true;

        if (parent[u] !== -1) {
            const p = parent[u];
            const dx = vertices[u].x - vertices[p].x;
            const dy = vertices[u].y - vertices[p].y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            edges.push({
                u,
                v: p,
                weight: dist
            });
        }

        for (let v = 0; v < n; v++) {
            if (!inMST[v]) {
                const dx = vertices[u].x - vertices[v].x;
                const dy = vertices[u].y - vertices[v].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < minEdge[v]) {
                    minEdge[v] = dist;
                    parent[v] = u;
                }
            }
        }
    }

    return edges;
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

// Walks the MST from the leftmost tooth, always stepping to the leftmost unvisited
// neighbor next. For an arch-shaped set of points this reproduces the true left-to-right
// tooth sequence far more reliably than sorting all teeth by X directly (which breaks down
// wherever the arch curves back on itself, e.g. the posterior teeth).
const computeMstOrder = (vertices, mstEdges, pca) => {
    const n = vertices.length;
    const adj = Array.from({ length: n }, () => []);
    mstEdges.forEach(edge => {
        adj[edge.u].push(edge.v);
        adj[edge.v].push(edge.u);
    });

    const getRotatedX = (pt) => {
        if (!pca) return pt.x;
        const cos = Math.cos(-pca.angle);
        const sin = Math.sin(-pca.angle);
        return (pt.x - pca.center.x) * cos - (pt.y - pca.center.y) * sin;
    };

    let startIdx = 0;
    let minRotX = Infinity;
    vertices.forEach((pt, i) => {
        const rx = getRotatedX(pt);
        if (rx < minRotX) {
            minRotX = rx;
            startIdx = i;
        }
    });

    const visited = new Array(n).fill(false);
    const order = [];
    const stack = [startIdx];
    while (stack.length > 0 && order.length < n) {
        const idx = stack.pop();
        if (visited[idx]) continue;
        visited[idx] = true;
        order.push(idx);
        const neighbors = adj[idx]
            .filter(nb => !visited[nb])
            .sort((a, b) => getRotatedX(vertices[b]) - getRotatedX(vertices[a]));
        neighbors.forEach(nb => stack.push(nb));
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

// Walks the MST order and splits it into groups wherever two consecutive teeth are not
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

    // Draw MST if checked
    if (mstCb.checked && !mstCb.disabled && predictions) {
        const mstStatus = document.getElementById('mst-status');
        if (predictions.length < 2) {
            if (mstStatus) {
                mstStatus.textContent = ' Requires 2+';
                mstStatus.style.color = '#f44336';
            }
        } else {
            if (mstStatus) {
                mstStatus.textContent = ' Active';
                mstStatus.style.color = '#4caf50';
            }

            // Compute centroids
            const vertices = predictions.map(pred => computeCentroid(pred.polygon));

            // Compute MST edges
            const mstEdges = computeMST(vertices);

            // Draw MST edges (dashed lines) - no distance labels for now
            mstEdges.forEach(edge => {
                const uPt = vertices[edge.u];
                const vPt = vertices[edge.v];

                ctx.beginPath();
                ctx.moveTo(uPt.x, uPt.y);
                ctx.lineTo(vPt.x, vPt.y);
                ctx.strokeStyle = '#00ff66'; // Vibrant green
                ctx.lineWidth = 2.5;
                ctx.setLineDash([5, 5]);
                ctx.stroke();
                ctx.setLineDash([]); // Reset
            });

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
        const mstStatus = document.getElementById('mst-status');
        if (mstStatus && !mstCb.disabled) mstStatus.textContent = '';
    }

    // Draw Segment Gap if checked - labels the distance for teeth that are NOT touching
    if (gapCb.checked && !gapCb.disabled && predictions) {
        const gapStatus = document.getElementById('gap-status');
        if (predictions.length < 2) {
            if (gapStatus) {
                gapStatus.textContent = ' Requires 2+';
                gapStatus.style.color = '#f44336';
            }
        } else {
            if (gapStatus) {
                gapStatus.textContent = ' Active';
                gapStatus.style.color = '#4caf50';
            }

            // Walk the tooth arch in MST order and compare each tooth's segment only to the
            // next one in that order - this follows the arch's actual curve (unlike a plain
            // X-sort, which breaks down where the arch bends back at the posterior teeth)
            const vertices = predictions.map(pred => computeCentroid(pred.polygon));
            const mstEdges = computeMST(vertices);
            const order = computeMstOrder(vertices, mstEdges, pca);
            const { W_pca, H_pca } = computeArchPcaBounds(predictions, pca, canvas.width, canvas.height);

            for (let i = 0; i < order.length - 1; i++) {
                const gap = computeGapNormWeight(predictions, order[i], order[i + 1], W_pca, H_pca, pca);
                if (!gap) continue;
                if (gap.normWeight <= TOUCH_NORM_WEIGHT) continue; // touching - not a gap

                const { ptA, ptB, normWeight } = gap;

                // Draw gap edge (dashed line) between the two closest polygon points
                ctx.beginPath();
                ctx.moveTo(ptA[0], ptA[1]);
                ctx.lineTo(ptB[0], ptB[1]);
                ctx.strokeStyle = '#ff3366'; // Neon pink - distinguishes gaps from MST's green
                ctx.lineWidth = 2.5;
                ctx.setLineDash([5, 5]);
                ctx.stroke();
                ctx.setLineDash([]); // Reset

                // Draw a two-line label at the middle of the edge: distance value, then "LOSS" below it
                const midX = (ptA[0] + ptB[0]) / 2;
                const midY = (ptA[1] + ptB[1]) / 2;
                const valText = normWeight.toFixed(2);
                const lossText = 'LOSS';

                ctx.font = 'bold 28px sans-serif';
                const valWidth = ctx.measureText(valText).width;
                const valHeight = 28;

                ctx.font = 'bold 20px sans-serif';
                const lossWidth = ctx.measureText(lossText).width;
                const lossHeight = 20;

                const padding = 12;
                const lineGap = 4;
                const boxWidth = Math.max(valWidth, lossWidth) + padding * 2;
                const boxHeight = padding * 2 + valHeight + lineGap + lossHeight;
                const boxX = midX - boxWidth / 2;
                const boxY = midY - boxHeight / 2;
                const valY = boxY + padding;
                const lossY = valY + valHeight + lineGap;

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

                // Distance value (top line)
                ctx.font = 'bold 28px sans-serif';
                ctx.fillStyle = '#ffffff';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(valText, midX, valY);

                // "LOSS" label (bottom line)
                ctx.font = 'bold 20px sans-serif';
                ctx.fillStyle = '#ff3366';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(lossText, midX, lossY);
            }
        }
    } else {
        const gapStatus = document.getElementById('gap-status');
        if (gapStatus && !gapCb.disabled) gapStatus.textContent = '';
    }

    // Draw FDI numbers if checked - labels teeth in MST arch order regardless of tooth count
    if (fdiCb.checked && !fdiCb.disabled && predictions) {
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
            const mstEdges = computeMST(vertices);
            const order = computeMstOrder(vertices, mstEdges, pca);
            const { W_pca, H_pca } = computeArchPcaBounds(predictions, pca, canvas.width, canvas.height);

            // Teeth that are touching share one group; a break between groups is a missing
            // tooth, so skip one FDI slot there before labeling the next group
            const groups = computeArchGroups(predictions, order, W_pca, H_pca, pca);

            const fdiLabels = isLower
                ? [46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36]
                : [16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26];

            let slot = 0;
            groups.forEach((group, groupIdx) => {
                if (groupIdx > 0) slot += 1; // skip the FDI slot left by the missing tooth

                group.forEach(vertexIdx => {
                    if (slot >= fdiLabels.length) {
                        slot += 1;
                        return;
                    }

                    const pred = predictions[vertexIdx];
                    const centerX = (pred.box[0] + pred.box[2]) / 2;
                    const centerY = (pred.box[1] + pred.box[3]) / 2;

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

                    ctx.font = 'bold 30px sans-serif';
                    ctx.fillStyle = '#00ffff';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(String(fdiLabels[slot]), centerX, centerY + 1);

                    slot += 1;
                });
            });
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

    // MST is enabled when Tooth Segmentation is checked
    mstCb.disabled = !yoloCb.checked;
    if (mstCb.disabled) {
        mstCb.checked = false;
        const mstStatus = document.getElementById('mst-status');
        if (mstStatus) mstStatus.textContent = '';
    }

    // Segment Gap is enabled when Tooth Segmentation is checked
    gapCb.disabled = !yoloCb.checked;
    if (gapCb.disabled) {
        gapCb.checked = false;
        const gapStatus = document.getElementById('gap-status');
        if (gapStatus) gapStatus.textContent = '';
    }

    // FDI is enabled when Tooth Segmentation is checked
    fdiCb.disabled = !yoloCb.checked;
    if (fdiCb.disabled) {
        fdiCb.checked = false;
        const fdiStatus = document.getElementById('fdi-status');
        if (fdiStatus) fdiStatus.textContent = '';
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

mstCb.addEventListener('change', () => {
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
