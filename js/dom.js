const darkCheckbox = document.getElementById('dark-mode-checkbox');
const fileInput = document.getElementById('oral-image-input');
const fileStatus = document.getElementById('oral-image-status');

// Master switch: when checked, forces every 2D VIEW checkbox on and locked
const autoAllCb = document.getElementById('auto-all-checkbox');

// 2D VIEW Checkboxes
const jawCb = document.getElementById('jaw-classification-checkbox');
const yoloCb = document.getElementById('yolov8-seg-checkbox');
const pcaCb = document.getElementById('pca-checkbox');
const archPathCb = document.getElementById('arch-path-checkbox');
const gapCb = document.getElementById('gap-checkbox');
const fdiCb = document.getElementById('fdi-checkbox');

let imageFiles = [];
let currentImageIndex = 0;
let currentObjectURL = null;
let currentImage = null;
const classificationCache = {};
const segmentationCache = {};
const toothAnalysisCache = {};

const canvas = document.getElementById('raw-canvas');
const ctx = canvas.getContext('2d');
const currentSliceDisplay = document.getElementById('current-slice');
const fileNameDisplay = document.getElementById('file-name');
