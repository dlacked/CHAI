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
// view - see drawFdiNumbers/redrawCanvas in render.js. One radio per pre-ResNet pipeline stage
// worth illustrating on its own (paper figures - see render.js's drawPcaJawOverlay/
// drawHeldKarpOrder/drawMirroringOverlay/drawCropBoxesOverlay for what each draws).
const vizOffRadio = document.getElementById('viz-off-radio');
const vizSegmentationRadio = document.getElementById('viz-segmentation-radio');
const vizPcaRadio = document.getElementById('viz-pca-radio');
const vizHeldKarpRadio = document.getElementById('viz-heldkarp-radio');
const vizMirroringRadio = document.getElementById('viz-mirroring-radio');
const vizCropRadio = document.getElementById('viz-crop-radio');
const getVizMode = () => {
    if (vizSegmentationRadio.checked) return 'segmentation';
    if (vizPcaRadio.checked) return 'pca';
    if (vizHeldKarpRadio.checked) return 'heldkarp';
    if (vizMirroringRadio.checked) return 'mirroring';
    if (vizCropRadio.checked) return 'crop';
    return 'off';
};

// MODELS radios: pick which model's own labeling is shown for the current image. Ghorbani/Yoon
// are self-contained full-image models (unlike CHAI's segmentation-then-classify pipeline) with
// no Arch Complexity model of their own - see js/render.js redrawCanvas's non-CHAI branch and
// server.py's /ghorbani_predict, /yoon_predict.
const modelChaiRadio = document.getElementById('model-chai-radio');
const modelGhorbaniRadio = document.getElementById('model-ghorbani-radio');
const modelYoonRadio = document.getElementById('model-yoon-radio');
const visualizationSidebarSection = document.getElementById('visualization-sidebar-section');
const getSelectedModel = () => {
    if (modelGhorbaniRadio.checked) return 'ghorbani';
    if (modelYoonRadio.checked) return 'yoon';
    return 'chai';
};

let imageFiles = [];
let currentImageIndex = 0;
let currentObjectURL = null;
let currentImage = null;
const segmentationCache = {};
const toothAnalysisCache = {};
const complexityCache = {};
const ghorbaniAnalysisCache = {};
const yoonAnalysisCache = {};

const canvas = document.getElementById('raw-canvas');
const ctx = canvas.getContext('2d');
const currentSliceDisplay = document.getElementById('current-slice');
const fileNameDisplay = document.getElementById('file-name');
