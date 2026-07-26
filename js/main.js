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

    // 이미지 바뀔 때마다 segmentation 새로 수행 - jaw classification은 이제 세그멘테이션
    // 결과로부터 redrawCanvas()가 즉시(동기적으로) 계산하므로 별도 호출이 필요 없다
    if (yoloCb.checked) {
        segmentImage(file);
    }
};

const handleSelectedFiles = (files, sourceInput) => {
    if (files.length === 0) {
        fileStatus.textContent = 'No directory selected';
        yoloCb.disabled = true;
        yoloCb.checked = false;
        updateCheckboxStates();
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

        yoloCb.disabled = true;
        yoloCb.checked = false;
        updateCheckboxStates();
        return;
    }

    currentImageIndex = 0;
    displayImage(currentImageIndex);

    // Enable checkbox and trigger segmentation
    yoloCb.disabled = false;
    segmentImage(imageFiles[0]);
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

// 2D VIEW Checkbox Dependency Logic
const updateCheckboxStates = () => {
    // Select All: force every feature checkbox checked and lock it so the user can't turn
    // anything off. Each level only locks once the one above it is actually checked, so the
    // chain still lights up progressively as classification/segmentation complete.
    if (autoAllCb.checked) {
        if (imageFiles.length > 0) {
            yoloCb.checked = true;
            yoloCb.disabled = true;
        }

        jawCb.checked = yoloCb.checked;
        jawCb.disabled = true;

        pcaCb.checked = jawCb.checked;
        pcaCb.disabled = true;

        archPathCb.checked = pcaCb.checked;
        archPathCb.disabled = true;

        gapCb.checked = archPathCb.checked;
        gapCb.disabled = true;

        // Unlike the stages above, FDI vs Palmer is a display-mode *choice*, not a pipeline
        // dependency, so Select All forces some notation on (defaulting to FDI the first time)
        // but leaves both radios enabled - the user can still switch between them.
        if (gapCb.checked && !fdiCb.checked && !palmerCb.checked) {
            fdiCb.checked = true;
        }
        const notationOn = gapCb.checked;
        fdiCb.disabled = !notationOn;
        palmerCb.disabled = !notationOn;
        if (!notationOn) {
            fdiCb.checked = false;
            palmerCb.checked = false;
        }

        return;
    }

    jawCb.disabled = !yoloCb.checked;
    if (jawCb.disabled) {
        jawCb.checked = false;
    }

    pcaCb.disabled = !jawCb.checked;
    if (pcaCb.disabled) pcaCb.checked = false;

    // Arch Path is enabled when PCA is checked
    archPathCb.disabled = !pcaCb.checked;
    if (archPathCb.disabled) {
        archPathCb.checked = false;
    }

    // Segment Gap is enabled when Arch Path is checked. With exactly 12 teeth detected there's
    // nothing to show (no possible gaps), but the checkbox itself stays checkable regardless.
    gapCb.disabled = !archPathCb.checked;
    if (gapCb.disabled) {
        gapCb.checked = false;
    }

    // FDI/Palmer notation choice is enabled when Segment Gap is checked
    const notationDisabled = !gapCb.checked;
    fdiCb.disabled = notationDisabled;
    palmerCb.disabled = notationDisabled;
    if (notationDisabled) {
        fdiCb.checked = false;
        palmerCb.checked = false;
        clearAnalysisResult();
        clearComplexityStatus();
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

yoloCb.addEventListener('change', () => {
    updateCheckboxStates();
    if (yoloCb.checked) {
        if (imageFiles.length > 0) {
            segmentImage(imageFiles[currentImageIndex]);
        }
    } else {
        redrawCanvas();
    }
});

jawCb.addEventListener('change', () => {
    updateCheckboxStates();
    redrawCanvas();
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

palmerCb.addEventListener('change', () => {
    updateCheckboxStates();
    redrawCanvas();
});

// Initialize states
yoloCb.disabled = true;
updateCheckboxStates();
