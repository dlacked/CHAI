// Bottom panel height added onto the canvas beyond the image's own natural height (see
// js/main.js's canvas sizing and redrawCanvas below) - blank strip reserved for drawBottomPanel's
// single-line Complexity/Missing summary, kept separate from the image so it never overlaps tooth
// badges or (more importantly) the exact gap/crowding region that text is describing.
const BOTTOM_PANEL_HEIGHT = 110;

// Draws a black badge with a '#' prefix + FDI tooth number (white border, white prefix, white
// number - all-white so the badge reads as neutral labeling rather than status/color-coding).
const drawFdiNumberBadge = (centerX, centerY, numText, prefix = '#') => {
    const boxWidth = 180;
    const boxHeight = 118;
    const rx = centerX - boxWidth / 2;
    const ry = centerY - boxHeight / 2;

    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') {
        ctx.roundRect(rx, ry, boxWidth, boxHeight, 16);
    } else {
        ctx.rect(rx, ry, boxWidth, boxHeight);
    }
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3.5;
    ctx.stroke();

    ctx.font = 'bold 72px sans-serif';
    const numWidth = ctx.measureText(numText).width;
    ctx.font = 'bold 38px sans-serif';
    const prefixWidth = prefix ? ctx.measureText(prefix).width : 0;

    const spacing = prefix ? 5 : 0;
    const totalWidth = prefixWidth + spacing + numWidth;
    const startX = centerX - totalWidth / 2;

    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#ffffff';

    // Draw the prefix in small font, if any
    if (prefix) {
        ctx.font = 'bold 38px sans-serif';
        ctx.fillText(prefix, startX, centerY + 1);
    }

    // Draw the number/notation in large font
    ctx.font = 'bold 72px sans-serif';
    ctx.fillText(numText, startX + prefixWidth + spacing, centerY + 1);
};

// Fills/outlines each detected tooth's segmented region once its FDI number is known - teeth
// without a number yet are left unfilled. Drawn before the number badges so the badges sit on
// top of the colored regions. Every tooth uses the same white fill/stroke rather than a
// per-number color - the FDI badge text is what identifies each tooth, not region color.
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

        ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
        ctx.fill();

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
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

// "PCA & Jaw" debug view: the two earliest pipeline stages, drawn together since jaw
// classification (classifyJawByCurvature) is computed directly from the PCA frame this draws -
// each tooth's centroid (yellow dot, the input classifyJawByCurvature/PCA both reduce every
// tooth to), the PCA principal axis itself (blue line through pca.center at pca.angle - the
// arch's estimated main direction, distinct from drawMirroringOverlay's perpendicular midline),
// and the resulting Upper/Lower label with the parabola's leading coefficient `a` (whose sign is
// the entire classification rule - see classifyJawByCurvature's docstring).
const drawPcaJawOverlay = (predictions, pca, jawResult) => {
    if (!predictions || predictions.length === 0 || !pca || !currentImage) return;

    predictions.forEach(pred => {
        const centroid = computeCentroid(pred.polygon);
        ctx.beginPath();
        ctx.arc(centroid.x, centroid.y, 9, 0, Math.PI * 2);
        ctx.fillStyle = '#ffeb3b';
        ctx.fill();
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.stroke();
    });

    const dx = Math.cos(pca.angle);
    const dy = Math.sin(pca.angle);
    const len = Math.max(currentImage.width, currentImage.height);
    ctx.beginPath();
    ctx.moveTo(pca.center.x - dx * len, pca.center.y - dy * len);
    ctx.lineTo(pca.center.x + dx * len, pca.center.y + dy * len);
    ctx.strokeStyle = '#2196f3';
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(pca.center.x, pca.center.y, 11, 0, Math.PI * 2);
    ctx.fillStyle = '#2196f3';
    ctx.fill();
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2;
    ctx.stroke();

    if (jawResult) {
        const text = `${jawResult.isUpper ? 'UPPER' : 'LOWER'}  (a = ${jawResult.a.toExponential(2)})`;
        ctx.font = 'bold 34px sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        const textWidth = ctx.measureText(text).width;
        const textX = Math.min(Math.max(pca.center.x + 24, 10), currentImage.width - textWidth - 30);
        const textY = Math.max(pca.center.y - 40, 40);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
        ctx.fillRect(textX - 12, textY - 26, textWidth + 24, 52);
        ctx.fillStyle = '#39ff14';
        ctx.fillText(text, textX, textY);
    }
};

// "Crop Boxes" debug view: the padded axis-aligned bounding box each tooth is actually cropped
// to for ResNet's visual branch - poly.min/max + 10px padding, in ORIGINAL pixel space (not the
// PCA-rotated frame computeToothMeta's geometry-branch bbox uses - this is a different bbox for
// a different branch). Matches ResNet/tooth/train.py's ToothDataset crop convention exactly, so
// what's drawn here is pixel-for-pixel what the visual branch actually sees.
const CROP_PAD = 10;
const drawCropBoxesOverlay = (predictions) => {
    if (!predictions || predictions.length === 0 || !currentImage) return;

    predictions.forEach(pred => {
        const poly = pred.polygon;
        if (!poly || poly.length === 0) return;

        let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
        poly.forEach(([x, y]) => {
            if (x < x1) x1 = x;
            if (x > x2) x2 = x;
            if (y < y1) y1 = y;
            if (y > y2) y2 = y;
        });
        x1 = Math.max(0, x1 - CROP_PAD);
        y1 = Math.max(0, y1 - CROP_PAD);
        x2 = Math.min(currentImage.width, x2 + CROP_PAD);
        y2 = Math.min(currentImage.height, y2 + CROP_PAD);

        ctx.strokeStyle = '#39ff14';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });
};

// Determines each detected tooth's FDI tens digit (quadrant) in the Holding state: uses the
// server-refined `mirror` flag (derived from raw digit occurrence sequence before Hungarian
// assignment) when available, falling back to PCA coordinate side (x_rot sign) if missing.
const computeHoldingTens = (order, cached, predictions, pca, isUpper) => {
    const tensByVertexIdx = new Array(predictions.length).fill(null);
    order.forEach((vertexIdx, i) => {
        // cached.rows is already in arch order (js/api.js runToothAnalysis: "rows[i]
        // corresponds to teeth[i], which was built from order[i]"), so a missing row here just
        // means this arch position has no prediction yet.
        const row = cached.rows[i];
        if (!row) return;

        if (typeof row.mirror === 'boolean') {
            tensByVertexIdx[vertexIdx] = isUpper ? (row.mirror ? 2 : 1) : (row.mirror ? 3 : 4);
        } else {
            const centroid = computeCentroid(predictions[vertexIdx].polygon);
            const { tx } = rotateToPcaFrame(centroid.x, centroid.y, pca);
            tensByVertexIdx[vertexIdx] = computeQuadrantTens(tx, isUpper);
        }
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

// Bottom panel: the blank strip redrawCanvas appends below the image's own height (see
// js/main.js's canvas sizing), one line: "Complexity: Class II    Missing: #41, #31" - kept
// below the image (not overlaid on it) so it never covers the exact gap/crowding region the
// text is describing, and never competes with tooth badges for the same pixels.
// Always draws the panel background (even with no complexity label or missing teeth) so the
// bottom strip doesn't flicker in and out as those values load in.
const drawBottomPanel = (complexityLabel, missingNumbers, imageWidth, imageHeight) => {
    const panelY = imageHeight;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
    ctx.fillRect(0, panelY, imageWidth, BOTTOM_PANEL_HEIGHT);

    const segments = [];
    if (complexityLabel) {
        segments.push({ text: `Complexity: Class ${complexityLabel}`, color: '#39ff14' });
    }
    if (missingNumbers && missingNumbers.length > 0) {
        const list = missingNumbers.map(n => `#${n}`).join(', ');
        segments.push({ text: `Missing: ${list}`, color: '#ff3366' });
    }
    if (segments.length === 0) return;

    const font = 'bold 40px sans-serif';
    const gap = 56;
    ctx.font = font;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    const widths = segments.map(s => ctx.measureText(s.text).width);
    const totalWidth = widths.reduce((a, b) => a + b, 0) + gap * (segments.length - 1);
    const centerY = panelY + BOTTOM_PANEL_HEIGHT / 2;
    let x = Math.max((imageWidth - totalWidth) / 2, 24);

    segments.forEach((s, i) => {
        ctx.fillStyle = s.color;
        ctx.fillText(s.text, x, centerY);
        x += widths[i] + gap;
    });
};

// Renders a non-CHAI model's own predictions directly - a plain box outline (these models give
// axis-aligned boxes, not CHAI's segmentation polygons) plus the same FDI badge style used
// everywhere else, so the two look like the same family of output despite the different pipeline
// underneath. No color-coding by confidence/correctness - matches drawSegmentation's own
// "the number badge identifies the tooth, not box color" convention.
const drawExternalModelBoxes = (predictions) => {
    if (!predictions || predictions.length === 0) return;

    predictions.forEach(p => {
        const [x1, y1, x2, y2] = p.box;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const centerX = (x1 + x2) / 2;
        const centerY = (y1 + y2) / 2;
        drawFdiNumberBadge(centerX, centerY, String(p.fdi_number));
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
    const selectedModel = getSelectedModel();

    // Ghorbani/Yoon are self-contained full-image models - no CHAI segmentation, no PCA/Held-Karp
    // arch ordering, no Arch Complexity model of their own (that's CHAI-specific geometry, not
    // part of either paper) - so this branch skips straight to their own /*_predict endpoint and
    // renders just the FDI-number boxes it returns, no bottom panel (see js/main.js's
    // resizeCanvasForCurrentModel, which only reserves BOTTOM_PANEL_HEIGHT for the 'chai' branch).
    if (selectedModel !== 'chai') {
        clearAnalysisResult();
        clearComplexityStatus();
        renderMissingStatus([]);
        if (!file) return;

        const cache = selectedModel === 'ghorbani' ? ghorbaniAnalysisCache : yoonAnalysisCache;
        const cached = cache[file.name];
        if (!cached) {
            (selectedModel === 'ghorbani' ? runGhorbaniAnalysis : runYoonAnalysis)(file);
            return;
        }
        if (cached.status === 'done') {
            drawExternalModelBoxes(cached.predictions);
        }
        return;
    }

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
    } else if (vizMode === 'pca') {
        drawPcaJawOverlay(predictions, pca, jawResult);
    } else if (vizMode === 'heldkarp') {
        drawHeldKarpOrder(predictions, pca, isUpper);
    } else if (vizMode === 'mirroring') {
        drawMirroringOverlay(predictions, pca);
    } else if (vizMode === 'crop') {
        drawCropBoxesOverlay(predictions);
    }

    const missingNumbers = computeMissingTeeth(predictions, pca, isUpper, isLower, hasClassification, file);
    renderMissingStatus(missingNumbers);

    // Drawn unconditionally (not gated on vizMode === 'off') - the panel lives outside the
    // image itself, so it doesn't compete with any of the debug overlays for the same pixels.
    const complexity = file ? complexityCache[file.name] : null;
    const complexityLabel = complexity && complexity.status === 'done'
        ? (COMPLEXITY_LABELS[complexity.classIdx] ?? null)
        : null;
    drawBottomPanel(complexityLabel, missingNumbers, currentImage.width, currentImage.height);
};
