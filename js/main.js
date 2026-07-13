const darkCheckbox = document.getElementById('dark-mode-checkbox');
const fileInput = document.getElementById('oral-image-input');
const fileStatus = document.getElementById('oral-image-status');

// 2D VIEW Checkboxes
const jawCb = document.getElementById('jaw-classification-checkbox');
const yoloCb = document.getElementById('yolov8-seg-checkbox');
const pcaCb = document.getElementById('pca-checkbox');
const mstCb = document.getElementById('mst-checkbox');
const fdiCb = document.getElementById('fdi-checkbox');


let imageFiles = [];
let currentImageIndex = 0;
let currentObjectURL = null;
let currentImage = null;
const classificationCache = {};
const segmentationCache = {};

const canvas = document.getElementById('raw-canvas');
const ctx = canvas.getContext('2d');
const currentSliceDisplay = document.getElementById('current-slice');
const fileNameDisplay = document.getElementById('file-name');

const applyMode = (dark) => {
    document.body.classList.toggle('light-mode', !dark);
    localStorage.setItem('darkMode', dark ? '1' : '0');
};

const calculatePCARotation = (predictions, canvasWidth, isUpper) => {
    if (!predictions || predictions.length === 0) return null;

    let leftPoints = [];
    let rightPoints = [];

    predictions.forEach(pred => {
        const poly = pred.polygon;
        if (!poly || poly.length === 0) return;

        poly.forEach(pt => {
            const x = pt[0];
            const y = pt[1];

            if (x < canvasWidth / 2) {
                leftPoints.push({ x, y });
            } else {
                rightPoints.push({ x, y });
            }
        });
    });

    if (leftPoints.length === 0 || rightPoints.length === 0) return null;

    // targetY 찾기 (lower의 경우 min y, upper의 경우 max y)
    let pLeft = null;
    leftPoints.forEach(pt => {
        if (!pLeft) {
            pLeft = pt;
            return;
        }

        const isTarget = isUpper ? (pt.y > pLeft.y) : (pt.y < pLeft.y);
        const isTie = Math.abs(pt.y - pLeft.y) < 1e-4;

        if (isTarget) {
            pLeft = pt;
        } else if (isTie) {
            // 좌측 영역 tie-breaker: 가장 작은 x (leftmost)
            if (pt.x < pLeft.x) {
                pLeft = pt;
            }
        }
    });

    let pRight = null;
    rightPoints.forEach(pt => {
        if (!pRight) {
            pRight = pt;
            return;
        }

        const isTarget = isUpper ? (pt.y > pRight.y) : (pt.y < pRight.y);
        const isTie = Math.abs(pt.y - pRight.y) < 1e-4;

        if (isTarget) {
            pRight = pt;
        } else if (isTie) {
            // 우측 영역 tie-breaker: 가장 큰 x (rightmost)
            if (pt.x > pRight.x) {
                pRight = pt;
            }
        }
    });

    if (!pLeft || !pRight) return null;

    const centerX = (pLeft.x + pRight.x) / 2;
    const centerY = (pLeft.y + pRight.y) / 2;
    const angle = Math.atan2(pRight.y - pLeft.y, pRight.x - pLeft.x);

    return {
        pLeft,
        pRight,
        center: { x: centerX, y: centerY },
        angle: angle
    };
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
    if (pcaCb.checked && !pcaCb.disabled && hasClassification && predictions) {
        pca = calculatePCARotation(predictions, canvas.width, isUpper);
    }

    ctx.save();
    if (pca) {
        // Rotate around the PCA center
        ctx.translate(pca.center.x, pca.center.y);
        ctx.rotate(-pca.angle);
        ctx.translate(-pca.center.x, -pca.center.y);
    }

    // Draw the main image
    ctx.drawImage(currentImage, 0, 0);

    // Draw YOLO Segmentation if checked
    if (yoloCb.checked && !yoloCb.disabled && predictions) {
        drawSegmentation(predictions);
    }

    // Draw PCA overlay if PCA is active
    if (pca) {
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
    }

    ctx.restore();
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

    // 이미지 바뀔 때마다 classification 및 segmentation 새로 수행
    if (!jawCb.disabled && jawCb.checked) {
        classifyJawImage(file);
    } else {
        document.getElementById('jaw-classification-result').textContent = '';
    }

    if (!yoloCb.disabled && yoloCb.checked) {
        segmentImage(file);
    } else {
        const resultSpan = document.getElementById('yolo-segmentation-result');
        if (resultSpan) resultSpan.textContent = '';
    }
};

const classifyJawImage = (file) => {
    const resultSpan = document.getElementById('jaw-classification-result');

    if (classificationCache[file.name]) {
        const data = classificationCache[file.name];
        const isUpper = data.class === 'upper';
        resultSpan.textContent = isUpper ? ' Maxilla (Upper Jaw)' : ' Mandible (Lower Jaw)';
        resultSpan.style.color = '#4caf50';
        jawCb.checked = true;
        updateCheckboxStates();
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

                jawCb.checked = true;
                updateCheckboxStates();
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
    yoloCb.disabled = !jawCb.checked;
    if (yoloCb.disabled) {
        yoloCb.checked = false;
        const resultSpan = document.getElementById('yolo-segmentation-result');
        if (resultSpan) resultSpan.textContent = '';
    }

    pcaCb.disabled = !yoloCb.checked;
    if (pcaCb.disabled) pcaCb.checked = false;

    mstCb.disabled = !pcaCb.checked;
    if (mstCb.disabled) mstCb.checked = false;

    fdiCb.disabled = !mstCb.checked;
    if (fdiCb.disabled) fdiCb.checked = false;
};

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
mstCb.addEventListener('change', updateCheckboxStates);

// Initialize states
jawCb.disabled = true;
updateCheckboxStates();
