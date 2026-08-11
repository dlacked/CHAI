// Draws a black badge with a '#' prefix + FDI tooth number (white prefix, bright green
// number) centered at the given point.
const drawFdiNumberBadge = (centerX, centerY, numText, prefix = '#') => {
    const boxWidth = 114;
    const boxHeight = 75;
    const rx = centerX - boxWidth / 2;
    const ry = centerY - boxHeight / 2;

    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') {
        ctx.roundRect(rx, ry, boxWidth, boxHeight, 12);
    } else {
        ctx.rect(rx, ry, boxWidth, boxHeight);
    }
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.fill();
    ctx.strokeStyle = '#39ff14';
    ctx.lineWidth = 2.7;
    ctx.stroke();

    ctx.font = 'bold 45px sans-serif';
    const numWidth = ctx.measureText(numText).width;
    ctx.font = 'bold 24px sans-serif';
    const prefixWidth = prefix ? ctx.measureText(prefix).width : 0;

    const spacing = prefix ? 3 : 0;
    const totalWidth = prefixWidth + spacing + numWidth;
    const startX = centerX - totalWidth / 2;

    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';

    // Draw the prefix in small font (white), if any
    if (prefix) {
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 24px sans-serif';
        ctx.fillText(prefix, startX, centerY + 1);
    }

    // Draw the number/notation in large font (bright green)
    ctx.fillStyle = '#39ff14';
    ctx.font = 'bold 45px sans-serif';
    ctx.fillText(numText, startX + prefixWidth + spacing, centerY + 1);
};

// Each tooth's color is derived from its FDI number itself (not its array index) via
// golden-ratio hue distribution, so tooth #36 is always the same color across frames/images
// instead of shifting with detection order
const getFdiToothColor = (fdiNumber) => {
    const hue = (fdiNumber * 137.5) % 360;
    return {
        fill: `hsla(${hue}, 70%, 50%, 0.45)`,
        stroke: `hsla(${hue}, 70%, 50%, 0.95)`
    };
};

// Fills/outlines each detected tooth's segmented region once its FDI number is known - teeth
// without a number yet are left unfilled. Drawn before the number badges so the badges sit on
// top of the colored regions.
const drawSegmentation = (predictions, fdiByIndex) => {
    if (!predictions || predictions.length === 0 || !fdiByIndex) return;

    predictions.forEach((pred, idx) => {
        const fdiNumber = fdiByIndex[idx];
        if (!fdiNumber) return;

        const poly = pred.polygon;
        if (!poly || poly.length === 0) return;

        ctx.beginPath();
        ctx.moveTo(poly[0][0], poly[0][1]);
        for (let i = 1; i < poly.length; i++) {
            ctx.lineTo(poly[i][0], poly[i][1]);
        }
        ctx.closePath();

        const { fill, stroke } = getFdiToothColor(fdiNumber);

        ctx.fillStyle = fill;
        ctx.fill();

        ctx.strokeStyle = stroke;
        ctx.lineWidth = 3;
        ctx.stroke();
    });
};

// "Segmentation" debug view: every detected polygon, regardless of FDI status, all in one
// translucent green. Unlike drawSegmentation, nothing is skipped just because FDI numbering
// hasn't resolved yet.
const drawRawSegmentation = (predictions) => {
    if (!predictions || predictions.length === 0) return;

    predictions.forEach(pred => {
        const poly = pred.polygon;
        if (!poly || poly.length === 0) return;

        ctx.beginPath();
        ctx.moveTo(poly[0][0], poly[0][1]);
        for (let i = 1; i < poly.length; i++) {
            ctx.lineTo(poly[i][0], poly[i][1]);
        }
        ctx.closePath();

        ctx.fillStyle = 'rgba(57, 255, 20, 0.35)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(57, 255, 20, 0.9)';
        ctx.lineWidth = 3;
        ctx.stroke();
    });
};

// "Held-Karp Order" debug view: the exact arch traversal computeArchOrder finds - a numbered
// badge (1, 2, 3, ...) at each tooth's centroid, joined by lines in that order, so the
// shortest-Hamiltonian-path result is visible directly instead of inferred from FDI badges.
const drawHeldKarpOrder = (predictions, pca, isUpper) => {
    if (!predictions || predictions.length === 0 || !pca) return;

    const vertices = predictions.map(pred => computeCentroid(pred.polygon));
    const order = computeArchOrder(vertices, pca, isUpper);
    if (order.length === 0) return;

    ctx.beginPath();
    order.forEach((vertexIdx, i) => {
        const { x, y } = vertices[vertexIdx];
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#39ff14';
    ctx.lineWidth = 4;
    ctx.stroke();

    order.forEach((vertexIdx, i) => {
        const { x, y } = vertices[vertexIdx];
        drawFdiNumberBadge(x, y, String(i + 1), '');
    });
};

// "Coords Mirroring" debug view: colors each tooth by whether computeToothMeta would mirror its
// x/theta (positive rotated-x about the PCA midline = mirrored, matching the Q2/Q3 rule), and
// draws that midline (x_rot = 0) itself so the mirror boundary is visible, not just its effect.
const drawMirroringOverlay = (predictions, pca) => {
    if (!predictions || predictions.length === 0 || !pca) return;

    predictions.forEach(pred => {
        const poly = pred.polygon;
        if (!poly || poly.length === 0) return;

        const centroid = computeCentroid(poly);
        const { tx } = rotateToPcaFrame(centroid.x, centroid.y, pca);
        const mirrored = tx >= 0;

        ctx.beginPath();
        ctx.moveTo(poly[0][0], poly[0][1]);
        for (let i = 1; i < poly.length; i++) {
            ctx.lineTo(poly[i][0], poly[i][1]);
        }
        ctx.closePath();

        ctx.fillStyle = mirrored ? 'rgba(255, 87, 34, 0.45)' : 'rgba(33, 150, 243, 0.45)';
        ctx.fill();
        ctx.strokeStyle = mirrored ? 'rgba(255, 87, 34, 0.95)' : 'rgba(33, 150, 243, 0.95)';
        ctx.lineWidth = 3;
        ctx.stroke();

        // Centroid marker - the exact point rotateToPcaFrame/computeQuadrantTens read tx from,
        // so the mirror decision above is visibly anchored to something on the canvas.
        ctx.beginPath();
        ctx.arc(centroid.x, centroid.y, 7, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.stroke();
    });

    // The midline every mirror decision above is made against: the line x_rot = 0, i.e. the
    // line through pca.center perpendicular to the principal axis.
    const dx = Math.cos(pca.angle + Math.PI / 2);
    const dy = Math.sin(pca.angle + Math.PI / 2);
    const len = Math.max(canvas.width, canvas.height);
    ctx.beginPath();
    ctx.moveTo(pca.center.x - dx * len, pca.center.y - dy * len);
    ctx.lineTo(pca.center.x + dx * len, pca.center.y + dy * len);
    ctx.strokeStyle = '#ffeb3b';
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 8]);
    ctx.stroke();
    ctx.setLineDash([]);
};

// Determines each detected tooth's FDI tens digit (quadrant) in the Holding state, purely from
// which side of the PCA midline (x_rot sign) the tooth's centroid falls on - see
// computeQuadrantTens. Independent of the predicted last digits, so a wrong/duplicated last
// digit from the ResNet classifier can never also throw off the quadrant assignment.
const computeHoldingTens = (order, cached, predictions, pca, isUpper) => {
    const tensByVertexIdx = new Array(predictions.length).fill(null);
    order.forEach((vertexIdx, i) => {
        // cached.rows is already in arch order (js/api.js runToothAnalysis: "rows[i]
        // corresponds to teeth[i], which was built from order[i]"), so a missing row here just
        // means this arch position has no prediction yet.
        if (!cached.rows[i]) return;

        const centroid = computeCentroid(predictions[vertexIdx].polygon);
        const { tx } = rotateToPcaFrame(centroid.x, centroid.y, pca);
        tensByVertexIdx[vertexIdx] = computeQuadrantTens(tx, isUpper);
    });

    return tensByVertexIdx;
};

// Pure computation of each prediction's FDI number (parallel array, null where still unknown),
// mirroring the ResNet branch in drawFdiNumbers but without any DOM writes or triggering
// the analysis call itself. Always sourced from the ResNet classifier (js/api.js
// runToothAnalysis) regardless of tooth count - pure-geometry slot assignment was dropped for
// being less accurate (see functions/graph/12tooth.py's comparison). Used by drawSegmentation
// (region coloring) and computeMissingTeeth (sidebar MISSING summary).
const computeFdiByIndexCore = (predictions, pca, isUpper, hasClassification, file) => {
    if (!predictions || !pca || !hasClassification) return null;

    const vertices = predictions.map(pred => computeCentroid(pred.polygon));
    const order = computeArchOrder(vertices, pca, isUpper);

    const cached = file ? toothAnalysisCache[file.name] : null;
    if (cached && cached.status === 'done') {
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

// Draws the predicted Arch Complexity class as a badge anchored in the canvas's bottom-right
// corner - a canvas-visible counterpart to the sidebar's #complexity-result text
// (renderComplexityStatus). Styled like drawFdiNumberBadge (black box, neon-green text) since
// it's a per-arch sibling of that per-tooth badge family. Returns the vertical space it took up
// (box height + gap) so drawMissingIndicator can stack its own badges above it.
const drawComplexityIndicator = (label) => {
    if (!label) return 0;

    const padding = 24;
    const boxHeight = 68;
    const boxGap = 14;
    const font = 'bold 40px sans-serif';
    const prefix = 'Complexity: ';
    const value = `Class ${label}`;

    ctx.font = font;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    const prefixWidth = ctx.measureText(prefix).width;
    const valueWidth = ctx.measureText(value).width;
    const boxWidth = prefixWidth + valueWidth + 36;
    const boxX = canvas.width - padding - boxWidth;
    const boxY = canvas.height - padding - boxHeight;

    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') {
        ctx.roundRect(boxX, boxY, boxWidth, boxHeight, 10);
    } else {
        ctx.rect(boxX, boxY, boxWidth, boxHeight);
    }
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.fill();
    ctx.strokeStyle = '#39ff14';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.font = font;
    const textY = boxY + boxHeight / 2 + 1;
    ctx.fillStyle = '#ffffff';
    ctx.fillText(prefix, boxX + 18, textY);
    ctx.fillStyle = '#39ff14';
    ctx.fillText(value, boxX + 18 + prefixWidth, textY);

    return boxHeight + boxGap;
};

// Draws each missing tooth's FDI number as a "MISSING #NN" badge in the canvas's bottom-right
// corner, stacked bottom-up starting above `bottomOffset` (the space drawComplexityIndicator
// already claimed) - a canvas-visible counterpart to the sidebar's #missing-result text
// (renderMissingStatus), for when the sidebar isn't in view. Background matches
// drawFdiNumberBadge's black box, so it reads as the same badge family as the FDI labels.
const drawMissingIndicator = (missingNumbers, bottomOffset = 0) => {
    if (!missingNumbers || missingNumbers.length === 0) return;

    const padding = 24;
    const boxHeight = 68;
    const boxGap = 14;
    const font = 'bold 44px sans-serif';

    ctx.font = font;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    // Bottom-up, so the first missing number ends up lowest (closest to the corner/offset).
    missingNumbers.forEach((n, i) => {
        const text = `MISSING #${n}`;
        const textWidth = ctx.measureText(text).width;
        const boxWidth = textWidth + 36;
        const boxX = canvas.width - padding - boxWidth;
        const boxY = canvas.height - padding - bottomOffset - boxHeight - i * (boxHeight + boxGap);

        ctx.beginPath();
        if (typeof ctx.roundRect === 'function') {
            ctx.roundRect(boxX, boxY, boxWidth, boxHeight, 10);
        } else {
            ctx.rect(boxX, boxY, boxWidth, boxHeight);
        }
        ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 51, 102, 0.25)';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.font = font;
        ctx.fillStyle = '#ff3366';
        ctx.fillText(text, boxX + 18, boxY + boxHeight / 2 + 1);
    });
};

// Labels teeth in arch order regardless of tooth count, and kicks off the per-tooth (ResNet+ViT)
// and per-arch (Arch Complexity) analysis calls. Only ever invoked when the pipeline is
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

    // Arch Complexity doesn't depend on FDI number assignment - it only needs
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

    // FDI numbers always come from the ResNet classifier (js/api.js runToothAnalysis and
    // server.py /tooth_predict), regardless of tooth count - pure-geometry slot assignment
    // was dropped for being less accurate (see functions/graph/12tooth.py's comparison). If
    // the analysis is still loading, just show a pending status - no debug overlay.
    const cached = file ? toothAnalysisCache[file.name] : null;

    // A debug overlay (Segmentation / Held-Karp / Mirroring) owns the canvas when active - the
    // analysis calls above still run so the sidebar stays live, but the FDI badges themselves
    // are left to redrawCanvas's overlay dispatch instead of being drawn here.
    if (cached && cached.status === 'done' && getVizMode() === 'off') {
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
        renderMissingStatus([]);
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

    // Colored tooth regions first, so the number badges drawn by drawFdiNumbers sit on top.
    // Skipped in favor of the debug overlay below whenever one is selected.
    const fdiByIndex = computeFdiByIndexCore(predictions, pca, isUpper, hasClassification, file);
    const vizMode = getVizMode();
    if (vizMode === 'off') {
        drawSegmentation(predictions, fdiByIndex);
    }

    // Still called unconditionally: this is what kicks off/keeps alive the ResNet+ViT and
    // Arch Complexity analysis calls (see drawFdiNumbers) regardless of which view is on
    // screen - only its own badge-drawing is gated on vizMode === 'off'.
    drawFdiNumbers(predictions, pca, isUpper, hasClassification, file);

    if (vizMode === 'segmentation') {
        drawRawSegmentation(predictions);
    } else if (vizMode === 'heldkarp') {
        drawHeldKarpOrder(predictions, pca, isUpper);
    } else if (vizMode === 'mirroring') {
        drawMirroringOverlay(predictions, pca);
    }

    const missingNumbers = computeMissingTeeth(predictions, pca, isUpper, isLower, hasClassification, file);
    renderMissingStatus(missingNumbers);
    if (vizMode === 'off') {
        const complexity = file ? complexityCache[file.name] : null;
        const complexityLabel = complexity && complexity.status === 'done'
            ? (COMPLEXITY_LABELS[complexity.classIdx] ?? null)
            : null;
        const offset = drawComplexityIndicator(complexityLabel);
        drawMissingIndicator(missingNumbers, offset);
    }
};
