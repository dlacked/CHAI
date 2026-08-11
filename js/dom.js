const darkCheckbox = document.getElementById('dark-mode-checkbox');
const fileInput = document.getElementById('oral-image-input');
const filesInput = document.getElementById('oral-image-files');
const fileStatus = document.getElementById('oral-image-status');

// Single on/off switch: On runs the full pipeline (segmentation through FDI numbering) and
// labels the image; Off just displays the raw image with no processing.
const pipelineOnRadio = document.getElementById('pipeline-on-radio');
const pipelineOffRadio = document.getElementById('pipeline-off-radio');
const isPipelineOn = () => pipelineOnRadio.checked;

// Debug overlay radios: mutually exclusive with each other, and with the normal FDI-badge
// view - see drawFdiNumbers/redrawCanvas in render.js.
const vizOffRadio = document.getElementById('viz-off-radio');
const vizSegmentationRadio = document.getElementById('viz-segmentation-radio');
const vizHeldKarpRadio = document.getElementById('viz-heldkarp-radio');
const vizMirroringRadio = document.getElementById('viz-mirroring-radio');
const getVizMode = () => {
    if (vizSegmentationRadio.checked) return 'segmentation';
    // vizHeldKarpRadio has no matching UI control anymore (index.html) - its radio-mode branch
    // in render.js is kept working (drawHeldKarpOrder etc.), just unreachable from the UI now.
    if (vizHeldKarpRadio && vizHeldKarpRadio.checked) return 'heldkarp';
    if (vizMirroringRadio.checked) return 'mirroring';
    return 'off';
};

let imageFiles = [];
let currentImageIndex = 0;
let currentObjectURL = null;
let currentImage = null;
const segmentationCache = {};
const toothAnalysisCache = {};
const complexityCache = {};

const canvas = document.getElementById('raw-canvas');
const ctx = canvas.getContext('2d');
const currentSliceDisplay = document.getElementById('current-slice');
const fileNameDisplay = document.getElementById('file-name');
