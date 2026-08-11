const applyMode = (dark) => {
    document.body.classList.toggle('light-mode', !dark);
    localStorage.setItem('darkMode', dark ? '1' : '0');
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

// Debug overlay radios don't fetch anything new - everything they draw comes from data
// already cached by the pipeline - so a redraw is all that's needed.
[vizOffRadio, vizSegmentationRadio, vizHeldKarpRadio, vizMirroringRadio].forEach(radio => {
    radio.addEventListener('change', () => {
        if (!radio.checked) return;
        redrawCanvas();
    });
});
