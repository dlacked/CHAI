const applyMode = (dark) => {
    document.body.classList.toggle('light-mode', !dark);
    localStorage.setItem('darkMode', dark ? '1' : '0');
};

// Extra height beyond the image's own size reserves the blank strip drawBottomPanel (js/render.js)
// lists the Complexity class and missing-tooth numbers in - only meaningful for CHAI's own
// pipeline (Ghorbani/Yoon have no Arch Complexity model, see js/render.js redrawCanvas's non-CHAI
// branch), so the panel - and the canvas space for it - only exists when CHAI is selected. Called
// both on image load and whenever the MODELS radio changes, so switching models on an
// already-loaded image resizes the canvas immediately rather than leaving stale blank space.
const resizeCanvasForCurrentModel = () => {
    if (!currentImage) return;
    canvas.width = currentImage.width;
    canvas.height = currentImage.height + (getSelectedModel() === 'chai' ? BOTTOM_PANEL_HEIGHT : 0);
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
        currentImage = img;
        resizeCanvasForCurrentModel();

        redrawCanvas();

        currentSliceDisplay.textContent = `Image: ${index + 1}/${imageFiles.length}`;
        fileNameDisplay.textContent = `File: ${file.name}`;
        fileStatus.textContent = `Loaded ${imageFiles.length} images.`;
    };
    img.src = currentObjectURL;

    // Pipeline Off: just show the image, no segmentation call
    if (isPipelineOn()) {
        segmentImage(file);
    }
};

const handleSelectedFiles = (files, sourceInput) => {
    if (files.length === 0) {
        fileStatus.textContent = 'No directory selected';
        return;
    }

    imageFiles = files
        .filter(file => {
            const ext = file.name.toLowerCase().split('.').pop();
            return ['jpg', 'jpeg', 'png'].includes(ext) || file.type.startsWith('image/');
        })
        .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }));

    if (imageFiles.length === 0) {
        fileStatus.textContent = 'No JPG, JPEG, or PNG images found in selection.';
        sourceInput.value = '';
        currentSliceDisplay.textContent = 'Image: 0/0';
        fileNameDisplay.textContent = 'File: Not Opened';
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
    }

    currentImageIndex = 0;
    displayImage(currentImageIndex);
};

fileInput.addEventListener('change', () => {
    handleSelectedFiles(Array.from(fileInput.files || []), fileInput);
});

filesInput.addEventListener('change', () => {
    handleSelectedFiles(Array.from(filesInput.files || []), filesInput);
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

// Pipeline On: run segmentImage (which itself short-circuits to the cached result and
// redraws immediately if this file was already segmented before Off was toggled on).
pipelineOnRadio.addEventListener('change', () => {
    if (!pipelineOnRadio.checked || imageFiles.length === 0) return;
    segmentImage(imageFiles[currentImageIndex]);
});

pipelineOffRadio.addEventListener('change', () => {
    if (!pipelineOffRadio.checked) return;
    redrawCanvas();
});

// Switching models resizes the canvas (see resizeCanvasForCurrentModel) before redrawing, since
// only CHAI reserves bottom-panel space - Ghorbani/Yoon's own analysis call (if not already
// cached for this file) is triggered from inside redrawCanvas's non-CHAI branch, same lazy-fetch
// pattern as segmentImage/runToothAnalysis.
[modelChaiRadio, modelGhorbaniRadio, modelYoonRadio].forEach(radio => {
    radio.addEventListener('change', () => {
        if (!radio.checked) return;
        resizeCanvasForCurrentModel();
        redrawCanvas();
    });
});

// Debug overlay radios don't fetch anything new - everything they draw comes from data
// already cached by the pipeline - so a redraw is all that's needed.
[vizOffRadio, vizSegmentationRadio, vizPcaRadio, vizHeldKarpRadio, vizMirroringRadio, vizCropRadio].forEach(radio => {
    radio.addEventListener('change', () => {
        if (!radio.checked) return;
        redrawCanvas();
    });
});
