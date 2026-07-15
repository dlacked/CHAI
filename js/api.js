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
