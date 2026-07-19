// Draws a black badge with a '#' + FDI number (white '#', bright green number) centered at the given point
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
    ctx.strokeStyle = '#39ff14';
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

    // Draw the number in large font (bright green)
    ctx.fillStyle = '#39ff14';
    ctx.font = 'bold 30px sans-serif';
    ctx.fillText(numText, startX + hashWidth + spacing, centerY + 1);
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
    if (!archPathCb.checked || !predictions) {
        const archPathStatus = document.getElementById('arch-path-status');
        if (archPathStatus) archPathStatus.textContent = '';
        return;
    }

    const archPathStatus = document.getElementById('arch-path-status');
    if (predictions.length < 2) {
        if (archPathStatus) {
            archPathStatus.textContent = ' Requires 2+';
            archPathStatus.style.color = '#f44336';
        }
        return;
    }

    if (archPathStatus) archPathStatus.textContent = '';

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

// Labels the distance for teeth that are NOT touching
const drawSegmentGap = (predictions, pca, isUpper, isLower, hasClassification) => {
    if (!gapCb.checked || !predictions) {
        const gapStatus = document.getElementById('gap-status');
        if (gapStatus) gapStatus.textContent = '';
        return;
    }

    const gapStatus = document.getElementById('gap-status');
    if (predictions.length < 2) {
        if (gapStatus) {
            gapStatus.textContent = ' Requires 2+';
            gapStatus.style.color = '#f44336';
        }
        return;
    }

    if (predictions.length >= 12) {
        if (gapStatus) {
            gapStatus.textContent = ' Loss (0)';
            gapStatus.style.color = '#4caf50';
        }
        return;
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

    let lossCount = 0;

    for (let i = 0; i < order.length - 1; i++) {
        const gap = computeGapNormWeight(predictions, order[i], order[i + 1], W_pca, H_pca, pca);
        if (!gap) continue;
        if (gap.normWeight <= TOUCH_NORM_WEIGHT) continue; // touching - not a gap

        lossCount += 1;

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
        const hashText = '#';
        const numText = missingLabel;
        const lossText = 'LOSS';
        const distText = `${gap.normWeight.toFixed(4)}`;

        // '#' is sized the same as the FDI badge's '#' (16px), number stays large
        ctx.font = 'bold 28px sans-serif';
        const numWidth = ctx.measureText(numText).width;
        ctx.font = 'bold 16px sans-serif';
        const hashWidth = ctx.measureText(hashText).width;
        const hashSpacing = 2;
        const valWidth = hashWidth + hashSpacing + numWidth;
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

        // Missing FDI number (top line, red) - small '#' + large number, vertically centered
        const valCenterY = valY + valHeight / 2;
        const valStartX = midX - valWidth / 2;
        ctx.fillStyle = '#ff3366';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';

        ctx.font = 'bold 16px sans-serif';
        ctx.fillText(hashText, valStartX, valCenterY);

        ctx.font = 'bold 28px sans-serif';
        ctx.fillText(numText, valStartX + hashWidth + hashSpacing, valCenterY);

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

    if (gapStatus) {
        gapStatus.textContent = ` Loss (${lossCount})`;
        gapStatus.style.color = lossCount > 0 ? '#ff3366' : '#4caf50';
    }
};

// Draws the raw coordinates/theta/greedy-sequence debug label for one tooth, used as the FDI
// fallback view when the detected teeth + gaps can't be reconstructed to exactly 12
const drawFdiDebugLabel = (pred, vertexIdx, order, slots, fdiLabels, pca) => {
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

    const archSeq = order.indexOf(vertexIdx);

    // 4. Build text lines
    const lines = [];
    lines.push(`Seq: ${archSeq}`);
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
};

// Labels teeth in nearest-neighbor arch order regardless of tooth count
const drawFdiNumbers = (predictions, pca, isUpper, isLower, hasClassification, file) => {
    if (!fdiCb.checked || !predictions || !pca) {
        const fdiStatus = document.getElementById('fdi-status');
        if (fdiStatus) fdiStatus.textContent = '';
        clearAnalysisResult();
        return;
    }

    const fdiStatus = document.getElementById('fdi-status');

    if (!hasClassification) {
        if (fdiStatus) {
            fdiStatus.textContent = ' Requires Jaw Classification';
            fdiStatus.style.color = '#f44336';
        }
        clearAnalysisResult();
        return;
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
        if (fdiStatus) fdiStatus.textContent = '';
        clearAnalysisResult();
        drawFdiLabels();
        return;
    }

    // Fewer than 12 teeth detected. Walk the arch order and count adjacent pairs that
    // are NOT touching (each gap = exactly one missing tooth by assumption). If detected
    // teeth + gaps reconstructs to exactly 12, the slot assignment is trustworthy - label
    // normally. Otherwise fall back to the ResNet Tooth model (Holding state).
    let lossCount = 0;
    for (let i = 0; i < order.length - 1; i++) {
        const gap = computeGapNormWeight(predictions, order[i], order[i + 1], W_pca, H_pca, pca);
        if (!gap) continue;
        if (gap.normWeight <= TOUCH_NORM_WEIGHT) continue; // touching - not a gap
        lossCount += 1;
    }
    const reconstructedTotal = predictions.length + lossCount;

    if (reconstructedTotal === 12) {
        if (fdiStatus) fdiStatus.textContent = '';
        clearAnalysisResult();
        drawFdiLabels();
        return;
    }

    // Holding: the 12-slot layout can't be trusted. The ResNet Tooth model fallback is
    // temporarily disabled pending retraining on the corrected PCA meta features (see
    // computeToothMeta) - always show the raw coordinates/theta/greedy-sequence debug labels
    // instead of drawing (potentially stale) ResNet predictions. runToothAnalysis still runs
    // in the background so results are cached and ready once the fallback is re-enabled.
    const jaw = isUpper ? 'upper' : 'lower';

    if (fdiStatus) {
        fdiStatus.textContent = ' Holding (Debug)';
        fdiStatus.style.color = '#f44336';
    }
    predictions.forEach((pred, vertexIdx) => {
        drawFdiDebugLabel(pred, vertexIdx, order, slots, fdiLabels, pca);
    });

    if (file) {
        runToothAnalysis(file, jaw, predictions, pca);
    }
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
        pca = calculatePCARotation(predictions);
    }

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
        drawSegmentation(predictions);
    }

    drawPcaOverlay(pca);
    drawArchPath(predictions, pca, isUpper);
    drawSegmentGap(predictions, pca, isUpper, isLower, hasClassification);
    drawFdiNumbers(predictions, pca, isUpper, isLower, hasClassification, file);

    ctx.restore();
};
