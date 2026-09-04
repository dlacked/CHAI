// Once segmentation succeeds, everything downstream (jaw classification/PCA/arch order/FDI
// numbering) is derived synchronously from the segmentation result inside redrawCanvas().
const onSegmented = (file) => {
    redrawCanvas();
};

const segmentImage = (file) => {
    if (segmentationCache[file.name]) {
        onSegmented(file);
        return;
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
                onSegmented(file);
            } else {
                console.error('Segmentation error:', data.error);
            }
        })
        .catch(error => {
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

const COMPLEXITY_LABELS = ['I', 'II', 'III'];

// Renders the Arch Complexity line into its own dedicated element
// (#complexity-result), deliberately separate from #analysis-result-list - that list gets
// wiped by clearAnalysisResult() whenever FDI reconstruction succeeds cleanly (see
// drawFdiNumbers/render.js), but this is independent of whether that reconstruction
// succeeded, so it must survive those clears.
const renderComplexityStatus = (file) => {
    const el = document.getElementById('complexity-result');
    if (!el) return;

    const complexity = file ? complexityCache[file.name] : null;
    if (!complexity) {
        el.innerHTML = '';
        return;
    }

    if (complexity.status === 'loading') {
        el.innerHTML = "Arch Complexity: <span class=\"analysis-complexity-loading\">Predicting...</span>";
        return;
    }
    if (complexity.status === 'error') {
        el.innerHTML = "Arch Complexity: <span class=\"analysis-complexity-error\">Error</span>";
        return;
    }
    const label = COMPLEXITY_LABELS[complexity.classIdx] ?? '?';
    el.innerHTML = `Arch Complexity: <span class="analysis-complexity-value">Class ${label}</span>`;
};

// Clears the Arch Complexity line - called whenever there are no teeth/jaw to classify
// at all (FDI unchecked, no predictions/PCA, or jaw classification missing).
const clearComplexityStatus = () => {
    const el = document.getElementById('complexity-result');
    if (el) el.innerHTML = '';
};

// Renders the MISSING: line into its own dedicated element (#missing-result), listing the
// missing teeth's raw FDI numbers with the same '#' prefix convention as drawFdiNumberBadge.
const renderMissingStatus = (missingNumbers) => {
    const el = document.getElementById('missing-result');
    if (!el) return;
    if (!missingNumbers || missingNumbers.length === 0) {
        el.textContent = '';
        return;
    }
    el.textContent = `MISSING: ${missingNumbers.map(n => `#${n}`).join(', ')}`;
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

// Clears the ANALYSIS RESULT per-tooth row list - called whenever the current image isn't in
// a Holding state (FDI unchecked, no classification yet, or the 12-slot reconstruction
// succeeded). Does NOT touch the Complexity Class line - see clearComplexityStatus.
const clearAnalysisResult = () => {
    const statusEl = document.getElementById('analysis-status');
    const listEl = document.getElementById('analysis-result-list');
    if (statusEl) statusEl.textContent = '';
    if (listEl) listEl.innerHTML = '';
};

// Sends the arch's per-tooth geometry (no crop needed - Transformer/complexity only takes
// [x1,y1,x2,y2] per tooth, no theta, see Transformer/complexity/model.py) to the complexity classifier
// and caches the predicted class (0=I, 1=II, 2=III). Runs independently of
// runToothAnalysis/tooth_predict so the Complexity Class line can show up even while the
// per-tooth FDI rows are still loading.
const runComplexityAnalysis = (file, jaw, teeth) => {
    const existing = complexityCache[file.name];
    if (existing && (existing.status === 'loading' || existing.status === 'done')) {
        renderComplexityStatus(file);
        return;
    }

    complexityCache[file.name] = { status: 'loading' };
    renderComplexityStatus(file);

    fetch('/complexity_predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            jaw,
            teeth: teeth.map(t => ({ x1: t.x1, y1: t.y1, x2: t.x2, y2: t.y2 }))
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                complexityCache[file.name] = { status: 'done', classIdx: data.class_idx, probs: data.probs };
            } else {
                complexityCache[file.name] = { status: 'error', error: data.error };
                console.error('Complexity analysis error:', data.error);
            }
            renderComplexityStatus(file);
        })
        .catch(error => {
            complexityCache[file.name] = { status: 'error', error: String(error) };
            console.error('Server offline or network error:', error);
            renderComplexityStatus(file);
        });
};

// Shared by runGhorbaniAnalysis/runYoonAnalysis: both endpoints are self-contained full-image
// models (no CHAI segmentation step, no per-tooth crop request) that take the raw file directly
// and return {box, fdi_number, confidence} per detection - see server.py's /ghorbani_predict,
// /yoon_predict. Caches by file.name like every other analysis cache in this file, and redraws
// once the result (or an error) lands so js/render.js's non-CHAI redrawCanvas branch can pick it
// up.
const runExternalModelAnalysis = (file, label, endpoint, cache) => {
    const existing = cache[file.name];
    if (existing && (existing.status === 'loading' || existing.status === 'done')) {
        redrawCanvas();
        return;
    }

    cache[file.name] = { status: 'loading' };
    redrawCanvas();

    const formData = new FormData();
    formData.append('image', file);

    fetch(endpoint, { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                cache[file.name] = { status: 'done', predictions: data.predictions };
            } else {
                cache[file.name] = { status: 'error', error: data.error };
                console.error(`${label} analysis error:`, data.error);
            }
            redrawCanvas();
        })
        .catch(error => {
            cache[file.name] = { status: 'error', error: String(error) };
            console.error('Server offline or network error:', error);
            redrawCanvas();
        });
};

const runGhorbaniAnalysis = (file) =>
    runExternalModelAnalysis(file, 'Ghorbani', '/ghorbani_predict', ghorbaniAnalysisCache);
const runYoonAnalysis = (file) =>
    runExternalModelAnalysis(file, 'Yoon', '/yoon_predict', yoonAnalysisCache);

// For every detected tooth, sends its crop + independent variables (x1, y1, x2, y2, theta -
// computed the same way as the GT feature extraction pipeline in functions/features/) to the
// ResNet Tooth model and caches the full 6-class probability vector for each tooth, ordered
// left-to-right along the arch (see computeArchOrder) - this is what powers both the ANALYSIS
// RESULT panel and the Holding-state FDI badges. `pca` is the shared PCA from
// calculatePCARotation() (redrawCanvas) - the same PCA used for the overlay, arch ordering,
// and the training CSVs (functions/features/theta.py calculate_pca_rotation). Triggers a
// redrawCanvas() once the result lands so the FDI badges can swap over from the raw debug view.
const runToothAnalysis = (file, jaw, predictions, pca) => {
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
    const order = computeArchOrder(vertices, pca, jaw === 'upper');
    // Baseline model (no PCA rotation anywhere, no theta - see computeToothMetaPooled's
    // docstring), not computeToothMeta - that one has no call sites left at all as of
    // 2026-08-31 (js/render.js drawFdiNumbers switched to computeToothMetaComplexity when the
    // Arch Complexity Transformer's own coordinate scheme moved off PCA too).
    const metas = predictions.map(pred => computeToothMetaPooled(pred, jaw === 'upper'));

    const teeth = order.map(idx => ({
        crop: cropToothImage(currentImage, predictions[idx].box),
        x1: metas[idx].x1,
        y1: metas[idx].y1,
        x2: metas[idx].x2,
        y2: metas[idx].y2,
        mirror: metas[idx].mirror
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
                // order[i] maps rows[i] back to its original prediction index.
                const rows = data.predictions.map(p => ({
                    probs: p.probs,
                    classIdx: p.class_idx,
                    mirror: p.mirror
                }));
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
