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

    // 이미지 바뀔 때마다 classification 새로 수행 - segmentation은 classification이 끝나면
    // onJawClassified()가 이어서 트리거하므로(항상 jawCb.checked일 때만 yoloCb.checked일 수
    // 있음) 여기서 별도로 호출하지 않는다
    if (jawCb.checked) {
        classifyJawImage(file);
    } else {
        document.getElementById('jaw-classification-result').textContent = '';
    }

    if (!yoloCb.checked) {
        const resultSpan = document.getElementById('yolo-segmentation-result');
        if (resultSpan) resultSpan.textContent = '';
    }
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
    // Select All: force every feature checkbox checked and lock it so the user can't turn
    // anything off. Each level only locks once the one above it is actually checked, so the
    // chain still lights up progressively as classification/segmentation complete.
    if (autoAllCb.checked) {
        if (imageFiles.length > 0) {
            jawCb.checked = true;
            jawCb.disabled = true;
        }

        yoloCb.checked = jawCb.checked;
        yoloCb.disabled = true;

        pcaCb.checked = yoloCb.checked;
        pcaCb.disabled = true;

        archPathCb.checked = pcaCb.checked;
        archPathCb.disabled = true;

        gapCb.checked = archPathCb.checked;
        gapCb.disabled = true;

        fdiCb.checked = gapCb.checked;
        fdiCb.disabled = true;

        const gapStatus = document.getElementById('gap-status');
        if (gapStatus && !gapCb.checked) gapStatus.textContent = '';

        const fdiStatus = document.getElementById('fdi-status');
        if (fdiStatus && !fdiCb.checked) fdiStatus.textContent = '';

        return;
    }

    yoloCb.disabled = !jawCb.checked;
    if (yoloCb.disabled) {
        yoloCb.checked = false;
        const resultSpan = document.getElementById('yolo-segmentation-result');
        if (resultSpan) resultSpan.textContent = '';
    }

    pcaCb.disabled = !yoloCb.checked;
    if (pcaCb.disabled) pcaCb.checked = false;

    // Arch Path is enabled when PCA is checked
    archPathCb.disabled = !pcaCb.checked;
    if (archPathCb.disabled) {
        archPathCb.checked = false;
        const archPathStatus = document.getElementById('arch-path-status');
        if (archPathStatus) archPathStatus.textContent = '';
    }

    const file = imageFiles[currentImageIndex];
    const predictions = file ? segmentationCache[file.name] : null;
    const is12Teeth = predictions && predictions.length === 12;

    // Segment Gap is enabled when Arch Path is checked. With exactly 12 teeth detected there's
    // nothing to show (no possible gaps), but the checkbox itself stays checkable regardless.
    gapCb.disabled = !archPathCb.checked;
    if (gapCb.disabled) {
        gapCb.checked = false;
        const gapStatus = document.getElementById('gap-status');
        if (gapStatus) gapStatus.textContent = '';
    } else {
        const gapStatus = document.getElementById('gap-status');
        if (gapStatus) {
            if (gapCb.checked) {
                gapStatus.textContent = ' Measuring...';
                gapStatus.style.color = 'rgba(255, 255, 255, 0.5)';
            } else if (is12Teeth) {
                gapStatus.textContent = ' Deactivated';
                gapStatus.style.color = '#f44336';
            } else {
                gapStatus.textContent = '';
            }
        }
    }

    // FDI is enabled when Segment Gap is checked
    fdiCb.disabled = !gapCb.checked;
    if (fdiCb.disabled) {
        fdiCb.checked = false;
        const fdiStatus = document.getElementById('fdi-status');
        if (fdiStatus) fdiStatus.textContent = '';
    }
};

autoAllCb.addEventListener('change', () => {
    updateCheckboxStates();
    if (yoloCb.checked && imageFiles.length > 0) {
        segmentImage(imageFiles[currentImageIndex]);
    } else {
        redrawCanvas();
    }
});

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

archPathCb.addEventListener('change', () => {
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
