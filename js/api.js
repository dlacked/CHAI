// Once jaw classification succeeds, check Jaw Classification and let updateCheckboxStates()
// cascade the rest (Select All forces everything on; otherwise Tooth Segmentation only runs
// if it's already checked, e.g. sticky from a previous image)
const onJawClassified = (file) => {
    jawCb.checked = true;
    updateCheckboxStates();
    if (yoloCb.checked) {
        segmentImage(file);
    }
};

const classifyJawImage = (file) => {
    const resultSpan = document.getElementById('jaw-classification-result');

    if (classificationCache[file.name]) {
        const data = classificationCache[file.name];
        const isUpper = data.class === 'upper';
        resultSpan.textContent = isUpper ? ' Maxilla (Upper Jaw)' : ' Mandible (Lower Jaw)';
        resultSpan.style.color = '#4caf50';
        onJawClassified(file);
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

                onJawClassified(file);
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

// Crops a tooth's bounding box out of the full image (10px padding, matching the crop used to
// build the ResNet Tooth model's training data - see ResNet/tooth/train.py ToothDataset) and
// returns it as a base64 PNG data URL for the /tooth_predict request
const cropToothImage = (image, box, padding = 10) => {
    const [x1, y1, x2, y2] = box;
    const cx1 = Math.max(0, Math.floor(x1 - padding));
    const cy1 = Math.max(0, Math.floor(y1 - padding));
    const cx2 = Math.min(image.width, Math.ceil(x2 + padding));
    const cy2 = Math.min(image.height, Math.ceil(y2 + padding));
    const w = Math.max(1, cx2 - cx1);
    const h = Math.max(1, cy2 - cy1);

    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = w;
    cropCanvas.height = h;
    cropCanvas.getContext('2d').drawImage(image, cx1, cy1, w, h, 0, 0, w, h);
    return cropCanvas.toDataURL('image/png');
};

// Renders the cached per-tooth probability vectors (left-to-right arch order) into the
// ANALYSIS RESULT sidebar panel - shown whenever a Holding-state prediction is cached for the
// current file, regardless of whether the FDI badges have swapped over yet
const renderAnalysisResult = (file) => {
    const listEl = document.getElementById('analysis-result-list');
    if (!listEl) return;

    const cached = file ? toothAnalysisCache[file.name] : null;
    if (!cached || cached.status !== 'done') {
        listEl.innerHTML = '';
        return;
    }

    listEl.innerHTML = cached.rows.map((row, i) => {
        const cells = row.probs.map((p, digit) => {
            const isTop = digit === row.classIdx;
            return `<span class="analysis-prob${isTop ? ' analysis-prob-top' : ''}">${digit + 1}: ${(p * 100).toFixed(1)}%</span>`;
        }).join('');
        return `<div class="analysis-row"><span class="analysis-row-idx">#${i + 1}</span><div class="analysis-row-probs">${cells}</div></div>`;
    }).join('');
};

// Clears the ANALYSIS RESULT panel - called whenever the current image isn't in a Holding
// state (FDI unchecked, no classification yet, or the 12-slot reconstruction succeeded)
const clearAnalysisResult = () => {
    const statusEl = document.getElementById('analysis-status');
    const listEl = document.getElementById('analysis-result-list');
    if (statusEl) statusEl.textContent = '';
    if (listEl) listEl.innerHTML = '';
};

// For every detected tooth, sends its crop + independent variables (x1, y1, x2, y2, theta -
// computed the same way as the GT feature extraction pipeline in functions/features/) to the
// ResNet Tooth model and caches the full 6-class probability vector for each tooth, ordered
// left-to-right along the arch (see computeArchOrder) - this is what powers both the ANALYSIS
// RESULT panel and the Holding-state FDI badges. `centroidPca` must be the true centroid PCA
// (calculateCentroidPCA), NOT the overlay heuristic pca from redrawCanvas - the two are
// different transforms and only the former matches training. Triggers a redrawCanvas() once
// the result lands so the FDI badges can swap over from the raw debug view.
const runToothAnalysis = (file, jaw, predictions, centroidPca) => {
    const statusEl = document.getElementById('analysis-status');
    const existing = toothAnalysisCache[file.name];
    if (existing && (existing.status === 'loading' || existing.status === 'done')) {
        if (statusEl && existing.status === 'done') {
            statusEl.textContent = ` Done (${existing.rows.length})`;
            statusEl.style.color = '#4caf50';
        }
        renderAnalysisResult(file);
        return;
    }

    toothAnalysisCache[file.name] = { status: 'loading' };
    if (statusEl) {
        statusEl.textContent = ' Predicting...';
        statusEl.style.color = 'rgba(255, 255, 255, 0.5)';
    }
    renderAnalysisResult(file);

    const vertices = predictions.map(pred => computeCentroid(pred.polygon));
    const order = computeArchOrder(vertices, centroidPca, jaw === 'upper');
    const metas = predictions.map(pred => computeToothMeta(pred, centroidPca));

    const teeth = order.map(idx => ({
        crop: cropToothImage(currentImage, predictions[idx].box),
        x1: metas[idx].x1,
        y1: metas[idx].y1,
        x2: metas[idx].x2,
        y2: metas[idx].y2,
        theta: metas[idx].theta
    }));

    fetch('/tooth_predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jaw, teeth })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // rows[i] corresponds to teeth[i], which was built from order[i] - so
                // order[i] maps rows[i] back to its original prediction index
                const rows = data.predictions.map(p => ({ probs: p.probs, classIdx: p.class_idx }));
                toothAnalysisCache[file.name] = { status: 'done', rows, order };
                if (statusEl) {
                    statusEl.textContent = ` Done (${rows.length})`;
                    statusEl.style.color = '#4caf50';
                }
            } else {
                toothAnalysisCache[file.name] = { status: 'error', error: data.error };
                if (statusEl) {
                    statusEl.textContent = ' Error';
                    statusEl.style.color = '#f44336';
                }
                console.error('Tooth analysis error:', data.error);
            }
            renderAnalysisResult(file);
            redrawCanvas();
        })
        .catch(error => {
            toothAnalysisCache[file.name] = { status: 'error', error: String(error) };
            if (statusEl) {
                statusEl.textContent = ' Offline';
                statusEl.style.color = '#f44336';
            }
            console.error('Server offline or network error:', error);
            renderAnalysisResult(file);
            redrawCanvas();
        });
};
