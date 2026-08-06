// Draws a black badge with a '#' prefix + FDI tooth number (white prefix, bright green
// number) centered at the given point.
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
// the analysis call itself. Always sourced from the ResNet+ViT arch transformer (js/api.js
// runToothAnalysis) regardless of tooth count - pure-geometry slot assignment was dropped for
// being less accurate (see functions/graph/12tooth.py's comparison). Used by computeMissingTeeth
// (sidebar LOSS summary).
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

// The raw FDI numbers of teeth judged missing from this jaw's 12-slot layout - runs purely
// off segmentation + jaw classification (and, in the Holding state, the arch
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

// Labels teeth in arch order regardless of tooth count, and kicks off the per-tooth (ResNet+ViT)
// and per-arch (Angle's Classification) analysis calls. Only ever invoked when the pipeline is
// On - see redrawCanvas.
const drawFdiNumbers = (predictions, pca, isUpper, hasClassification, file) => {
    if (!predictions || !pca || !hasClassification) {
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
            drawFdiNumberBadge(centerX, centerY, String(fdiNumber));
        });
    }

    if (file) {
        runToothAnalysis(file, jaw, predictions, pca);
    }
};

const redrawCanvas = () => {
    if (!currentImage) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImage, 0, 0);

    // Pipeline Off: just the raw image, no processing and no sidebar info.
    if (!isPipelineOn()) {
        clearAnalysisResult();
        clearComplexityStatus();
        renderLossStatus([]);
        return;
    }

    const file = imageFiles[currentImageIndex];
    const predictions = file ? segmentationCache[file.name] : null;

    // Jaw (upper/lower) is determined geometrically from the arch's curvature the moment
    // segmentation results are available - no /classify round trip (see
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

    drawFdiNumbers(predictions, pca, isUpper, hasClassification, file);

    const missingNumbers = computeMissingTeeth(predictions, pca, isUpper, isLower, hasClassification, file);
    renderLossStatus(missingNumbers);
};
