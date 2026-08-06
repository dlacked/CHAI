// Draws a black badge with an optional '#' prefix + tooth number/notation (white prefix,
// bright green number) centered at the given point. prefix defaults to '#' for FDI - pass ''
// for Palmer notation, which has no such convention (see formatToothLabel).
const drawFdiNumberBadge = (centerX, centerY, numText, prefix = '#') => {
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
    ctx.strokeStyle = '#39ff14';
    ctx.lineWidth = 1.8;
    ctx.stroke();

    ctx.font = 'bold 30px sans-serif';
    const numWidth = ctx.measureText(numText).width;
    ctx.font = 'bold 16px sans-serif';
    const prefixWidth = prefix ? ctx.measureText(prefix).width : 0;

    const spacing = prefix ? 2 : 0;
    const totalWidth = prefixWidth + spacing + numWidth;
    const startX = centerX - totalWidth / 2;

    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';

    // Draw the prefix in small font (white), if any
    if (prefix) {
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 16px sans-serif';
        ctx.fillText(prefix, startX, centerY + 1);
    }

    // Draw the number/notation in large font (bright green)
    ctx.fillStyle = '#39ff14';
    ctx.font = 'bold 30px sans-serif';
    ctx.fillText(numText, startX + prefixWidth + spacing, centerY + 1);
};

// Palmer notation quadrant abbreviations, keyed by FDI quadrant digit (1=UL, 2=UR, 3=LR,
// 4=LL - same quadrant order as every fdiLabels array in this file). R/L here match this
// app's on-screen left/right, not FDI's patient-relative convention - swapped from the
// textbook FDI-quadrant-1-is-patient's-upper-right mapping per user report.
const PALMER_QUADRANT_LABELS = { 1: 'UL', 2: 'UR', 3: 'LR', 4: 'LL' };

// Formats a full 2-digit FDI number (e.g. 46) as either plain FDI text or Palmer quadrant
// abbreviation + tooth number (e.g. "UR6"), depending on which radio (fdiCb/palmerCb) is
// currently selected.
const formatToothLabel = (fdiNumber, notation) => {
    if (notation !== 'palmer') return String(fdiNumber);
    const quadrant = Math.floor(fdiNumber / 10);
    const digit = fdiNumber % 10;
    const label = PALMER_QUADRANT_LABELS[quadrant];
    return label ? `${label}${digit}` : String(fdiNumber);
};

// Default color for a tooth whose FDI number hasn't been determined yet: transparent green
const UNLABELED_TOOTH_FILL = 'rgba(0, 200, 83, 0.18)';
const UNLABELED_TOOTH_STROKE = 'rgba(0, 200, 83, 0.55)';

// Once a tooth's FDI number is known, its color is derived from the number itself (not its
// array index) via golden-ratio hue distribution, so tooth #36 is always the same color
// across frames/images instead of shifting with detection order
const getFdiToothColor = (fdiNumber) => {
    const hue = (fdiNumber * 137.5) % 360;
    return {
        fill: `hsla(${hue}, 70%, 50%, 0.45)`,
        stroke: `hsla(${hue}, 70%, 50%, 0.95)`
    };
};

const drawSegmentation = (predictions, fdiByIndex) => {
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

        const fdiNumber = fdiByIndex ? fdiByIndex[idx] : null;
        const { fill, stroke } = fdiNumber
            ? getFdiToothColor(fdiNumber)
            : { fill: UNLABELED_TOOTH_FILL, stroke: UNLABELED_TOOTH_STROKE };

        ctx.fillStyle = fill;
        ctx.fill();

        ctx.strokeStyle = stroke;
        ctx.lineWidth = 3;
        ctx.stroke();
    });
};

// Determines each detected tooth's FDI tens digit (quadrant) in the Holding state, using the
// actual sequence of predicted last digits rather than geometry alone. A full arch normally
// has each last digit (1-6) appear twice - once per quadrant - so whichever arch-order
// position hits a given last digit FIRST gets that jaw's first quadrant (4 lower / 1 upper),
// and its second occurrence gets the second quadrant (3 lower / 2 upper). Only a last digit
// that appears just once (no second occurrence to disambiguate against) falls back to the
// pure geometric split (computeQuadrantTens).
const computeHoldingTens = (order, cached, predictions, pca, isUpper) => {
    const firstTens = isUpper ? 1 : 4;
    const secondTens = isUpper ? 2 : 3;

    // cached.rows is already in arch order (js/api.js runToothAnalysis: "rows[i] corresponds
    // to teeth[i], which was built from order[i]"), so position i's last digit is rows[i].
    const digitsInOrder = order.map((_, i) => {
        const row = cached.rows[i];
        return row ? row.classIdx + 1 : null;
    });

    const totalCount = {};
    digitsInOrder.forEach(digit => {
        if (digit !== null) totalCount[digit] = (totalCount[digit] || 0) + 1;
    });

    const seenSoFar = {};
    const tensByVertexIdx = new Array(predictions.length).fill(null);
    order.forEach((vertexIdx, i) => {
        const digit = digitsInOrder[i];
        if (digit === null) return;

        seenSoFar[digit] = (seenSoFar[digit] || 0) + 1;

        if (totalCount[digit] >= 2) {
            tensByVertexIdx[vertexIdx] = seenSoFar[digit] === 1 ? firstTens : secondTens;
        } else {
            const centroid = computeCentroid(predictions[vertexIdx].polygon);
            const { tx } = rotateToPcaFrame(centroid.x, centroid.y, pca);
            tensByVertexIdx[vertexIdx] = computeQuadrantTens(tx, isUpper);
        }
    });

    return tensByVertexIdx;
};

// Pure computation of each prediction's FDI number (parallel array, null where still unknown),
// mirroring the ResNet+ViT branch in drawFdiNumbers but without any DOM writes or triggering
// the analysis call itself - used purely to color drawSegmentation's polygons as soon as a
// number is available. Always sourced from the ResNet+ViT arch transformer (js/api.js
// runToothAnalysis) regardless of tooth count - pure-geometry slot assignment was dropped for
// being less accurate (see functions/graph/12tooth.py's comparison).
// Core FDI-per-detected-tooth computation, independent of the FDI/Palmer checkboxes - shared
// by computeFdiByIndex (segmentation coloring, gated on those checkboxes) and
// computeMissingTeeth (sidebar LOSS summary, which must NOT depend on them - see user request).
const computeFdiByIndexCore = (predictions, pca, isUpper, hasClassification, file) => {
    if (!predictions || !pca || !hasClassification) return null;

    const vertices = predictions.map(pred => computeCentroid(pred.polygon));
    const order = computeArchOrder(vertices, pca, isUpper);

    const cached = file ? toothAnalysisCache[file.name] : null;
    if (cached && cached.status === 'done' && cached.refined) {
        const tensByVertexIdx = computeHoldingTens(order, cached, predictions, pca, isUpper);
        const result = new Array(predictions.length).fill(null);
        predictions.forEach((pred, vertexIdx) => {
            const archSeq = order.indexOf(vertexIdx);
            const row = cached.rows[archSeq];
            if (!row) return;
            result[vertexIdx] = tensByVertexIdx[vertexIdx] * 10 + (row.classIdx + 1);
        });
        return result;
    }

    return null;
};

const computeFdiByIndex = (predictions, pca, isUpper, hasClassification, file) => {
    if (!(fdiCb.checked || palmerCb.checked)) return null;
    return computeFdiByIndexCore(predictions, pca, isUpper, hasClassification, file);
};

// The raw FDI numbers (not display text - see formatToothLabel for that) of teeth judged
// missing from this jaw's 12-slot layout, independent of any 2D VIEW checkbox (FDI, Palmer) -
// runs purely off segmentation + jaw classification (and, in the Holding state, the arch
// transformer's per-tooth predictions via computeFdiByIndexCore) whenever fewer than 12 teeth
// are detected.
const computeMissingTeeth = (predictions, pca, isUpper, isLower, hasClassification, file) => {
    if (!predictions || predictions.length >= 12) return [];

    const fdiByIndex = computeFdiByIndexCore(predictions, pca, isUpper, hasClassification, file);
    if (!fdiByIndex) return [];

    const fdiLabels = isLower
        ? [46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36]
        : [16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26];
    const found = new Set(fdiByIndex.filter(n => n !== null));
    return fdiLabels.filter(n => !found.has(n));
};

const drawPcaOverlay = (pca) => {
    if (!pca || !pcaCb.checked) return;

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
};

const drawArchPath = (predictions, pca, isUpper) => {
    if (!archPathCb.checked || !predictions) return;
    if (predictions.length < 2) return;

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
};

// Labels teeth in nearest-neighbor arch order regardless of tooth count
const drawFdiNumbers = (predictions, pca, isUpper, hasClassification, file) => {
    if (!(fdiCb.checked || palmerCb.checked) || !predictions || !pca) {
        clearAnalysisResult();
        clearComplexityStatus();
        return;
    }

    if (!hasClassification) {
        clearAnalysisResult();
        clearComplexityStatus();
        return;
    }

    const jaw = isUpper ? 'upper' : 'lower';
    const vertices = predictions.map(pred => computeCentroid(pred.polygon));
    const order = computeArchOrder(vertices, pca, isUpper);

    // Angle's Classification doesn't depend on FDI number assignment - it only needs
    // per-tooth geometry - so it's kicked off unconditionally here rather than only
    // alongside the tooth analysis call below (see runComplexityAnalysis / js/api.js).
    if (file) {
        const metas = predictions.map(pred => computeToothMeta(pred, pca));
        const complexityTeeth = order.map(idx => ({
            x1: metas[idx].x1,
            y1: metas[idx].y1,
            x2: metas[idx].x2,
            y2: metas[idx].y2,
            theta: metas[idx].theta
        }));
        runComplexityAnalysis(file, jaw, complexityTeeth);
    }

    const notation = palmerCb.checked ? 'palmer' : 'fdi';

    // FDI numbers always come from the ResNet+ViT arch transformer (js/api.js
    // runToothAnalysis and server.py /tooth_predict), regardless of tooth count -
    // pure-geometry slot assignment was dropped for being less accurate (see
    // functions/graph/12tooth.py's comparison). If the arch model isn't trained yet, or
    // the analysis is still loading, just show a pending status - no debug overlay.
    const cached = file ? toothAnalysisCache[file.name] : null;

    if (cached && cached.status === 'done' && cached.refined) {
        const tensByVertexIdx = computeHoldingTens(order, cached, predictions, pca, isUpper);
        predictions.forEach((pred, vertexIdx) => {
            const archSeq = order.indexOf(vertexIdx);
            const row = cached.rows[archSeq];
            if (!row) return;

            const fdiNumber = tensByVertexIdx[vertexIdx] * 10 + (row.classIdx + 1);

            const centerX = (pred.box[0] + pred.box[2]) / 2;
            const centerY = (pred.box[1] + pred.box[3]) / 2;
            drawFdiNumberBadge(centerX, centerY, formatToothLabel(fdiNumber, notation), notation === 'palmer' ? '' : '#');
        });
    }

    if (file) {
        runToothAnalysis(file, jaw, predictions, pca);
    }
};

const redrawCanvas = () => {
    if (!currentImage) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const file = imageFiles[currentImageIndex];
    const predictions = file ? segmentationCache[file.name] : null;

    // Jaw (upper/lower) is now determined geometrically from the arch's curvature the moment
    // segmentation results are available - no more /classify round trip (see
    // classifyJawByCurvature in js/geometry.js).
    let pca = null;
    let jawResult = null;
    if (predictions) {
        pca = calculatePCARotation(predictions);
        jawResult = pca ? classifyJawByCurvature(predictions, pca) : null;
    }
    const isUpper = !!(jawResult && jawResult.isUpper);
    const isLower = !!(jawResult && !jawResult.isUpper);
    const hasClassification = !!jawResult;

    ctx.save();
    if (pca && pcaCb.checked) {
        // Rotate around the PCA center
        ctx.translate(pca.center.x, pca.center.y);
        ctx.rotate(-pca.angle);
        ctx.translate(-pca.center.x, -pca.center.y);
    }

    // Draw the main image
    ctx.drawImage(currentImage, 0, 0);

    // Draw YOLO Segmentation if checked
    if (yoloCb.checked && predictions) {
        const fdiByIndex = computeFdiByIndex(predictions, pca, isUpper, hasClassification, file);
        drawSegmentation(predictions, fdiByIndex);
    }

    drawPcaOverlay(pca);
    drawArchPath(predictions, pca, isUpper);
    drawFdiNumbers(predictions, pca, isUpper, hasClassification, file);

    // Independent of FDI/Palmer being checked - see computeMissingTeeth. Still formatted per
    // whichever notation is selected (defaulting to FDI), same as the badges.
    const lossNotation = palmerCb.checked ? 'palmer' : 'fdi';
    const missingNumbers = computeMissingTeeth(predictions, pca, isUpper, isLower, hasClassification, file);
    renderLossStatus(missingNumbers.map(n => formatToothLabel(n, lossNotation)), lossNotation);

    ctx.restore();
};
