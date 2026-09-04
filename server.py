import io
import os
import sys
import base64
import importlib.util
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load config path if needed, or define locally for simplicity
ROOT = Path(__file__).resolve().parent

# Skips loading the comparison-paper reproductions (Ghorbani's YOLO detect+classify, Yoon's
# ResNet-101 Cascade R-CNN) - useful when a training run (e.g. Yoon's) already has the GPU mostly
# full and this server is only needed to check something CHAI-only (like the Tight Crop / PCA-
# Aligned Crop rendering), since neither reproduction is involved in CHAI's own pipeline at all.
SKIP_COMPARISON_MODELS = os.environ.get("SKIP_COMPARISON_MODELS") == "1"

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tooth classifier model(s)
TOOTH_MODEL_DIR = ROOT / "ResNet" / "tooth" / "model"
TOOTH_MODEL_PATHS = {
    "lower": TOOTH_MODEL_DIR / "lower_best.pth",
    "upper": TOOTH_MODEL_DIR / "upper_best.pth"
}

# Load ToothPositionClassifier from ResNet/tooth/train.py itself (single source of truth,
# same dynamic-import approach used for the Transformer models below) instead of keeping a
# hand-copied second definition here that could silently drift out of sync with training.
RESNET_TOOTH_TRAIN_PATH = ROOT / "ResNet" / "tooth" / "train.py"
_resnet_tooth_train_spec = importlib.util.spec_from_file_location("resnet_tooth_train", str(RESNET_TOOTH_TRAIN_PATH))
_resnet_tooth_train_module = importlib.util.module_from_spec(_resnet_tooth_train_spec)
_resnet_tooth_train_spec.loader.exec_module(_resnet_tooth_train_module)
ToothPositionClassifier = _resnet_tooth_train_module.ToothPositionClassifier


def build_tooth_model(num_classes=6):
    # pretrained=False: every weight gets overwritten by load_state_dict right below, so
    # fetching ImageNet-pretrained weights first would just be wasted network/disk I/O.
    return ToothPositionClassifier(num_classes=num_classes, pretrained=False)


tooth_models = {}
for jaw_name, model_path in TOOTH_MODEL_PATHS.items():
    if model_path.exists():
        print(f"Loading tooth model for {jaw_name} from {model_path}...")
        tooth_model = build_tooth_model(num_classes=6)
        tooth_model.load_state_dict(torch.load(model_path, map_location=device))
        tooth_model.to(device)
        tooth_model.eval()
        tooth_models[jaw_name] = tooth_model
    else:
        print(f"Tooth model weights not found for {jaw_name}: {model_path}")

# Pooled tooth model (one model for both jaws, no PCA rotation, no theta - see
# functions/features/coords_baseline.py / ResNet/tooth/train_baseline.py) - what /tooth_predict
# actually uses now. tooth_models above is left loaded (unused by that route) so reverting to the
# separate-per-jaw models is a one-line change if this doesn't pan out.
BASELINE_MODEL_PATH = ROOT / "ResNet" / "tooth" / "model_baseline" / "best.pth"
RESNET_TOOTH_TRAIN_BASELINE_PATH = ROOT / "ResNet" / "tooth" / "train_baseline.py"
_resnet_tooth_train_baseline_spec = importlib.util.spec_from_file_location(
    "resnet_tooth_train_baseline", str(RESNET_TOOTH_TRAIN_BASELINE_PATH))
_resnet_tooth_train_baseline_module = importlib.util.module_from_spec(_resnet_tooth_train_baseline_spec)
_resnet_tooth_train_baseline_spec.loader.exec_module(_resnet_tooth_train_baseline_module)
ToothPositionClassifierNoTheta = _resnet_tooth_train_baseline_module.ToothPositionClassifierNoTheta

baseline_tooth_model = None
if BASELINE_MODEL_PATH.exists():
    print(f"Loading baseline tooth model from {BASELINE_MODEL_PATH}...")
    baseline_tooth_model = ToothPositionClassifierNoTheta(num_classes=6, pretrained=False)
    baseline_tooth_model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=device))
    baseline_tooth_model.to(device)
    baseline_tooth_model.eval()
else:
    print(f"Baseline tooth model weights not found: {BASELINE_MODEL_PATH}")

# Load the arch-level Transformer that predicts an arch's overall complexity class (I/II/III)
# from per-tooth geometry alone - no image or ResNet features involved (see
# Transformer/complexity/model.py). Same dynamic-import approach as the ArchToothTransformer above.
# ONE model as of 2026-08-31 (was per-jaw {lower,upper}_best.pth before - promoted to a single
# pooled model, held-out test confirmed pooling doesn't hurt accuracy, see project notes) - jaw no
# longer selects a different model, just gets validated as a sane value in /complexity_predict.
COMPLEXITY_MODEL_PATH = ROOT / "Transformer" / "complexity" / "model.py"
COMPLEXITY_MODEL_WEIGHTS = ROOT / "Transformer" / "complexity" / "model" / "best.pth"
COMPLEXITY_MAX_TEETH = 12  # matches Transformer/complexity/build_dataset.py's MAX_TEETH_PER_ARCH
COMPLEXITY_GEOM_DIM = 4  # no-theta as of 2026-08-31 - see Transformer/complexity/train.py

complexity_model = None
if COMPLEXITY_MODEL_PATH.exists():
    spec = importlib.util.spec_from_file_location("vit_complexity_model", str(COMPLEXITY_MODEL_PATH))
    _vit_complexity_model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_vit_complexity_model_module)
    ArchComplexityTransformer = _vit_complexity_model_module.ArchComplexityTransformer

    if COMPLEXITY_MODEL_WEIGHTS.exists():
        print(f"Loading pooled complexity transformer from {COMPLEXITY_MODEL_WEIGHTS}...")
        complexity_model = ArchComplexityTransformer(geom_dim=COMPLEXITY_GEOM_DIM, num_classes=3)
        complexity_model.load_state_dict(torch.load(COMPLEXITY_MODEL_WEIGHTS, map_location=device))
        complexity_model.to(device)
        complexity_model.eval()
    else:
        print(f"Complexity transformer weights not found: {COMPLEXITY_MODEL_WEIGHTS}")
else:
    print(f"Transformer/complexity/model.py not found at {COMPLEXITY_MODEL_PATH}; complexity classification disabled.")

# Initialize YOLO segmentation model
print("Initializing YOLO segmentation model...")
try:
    from ultralytics import YOLO
    YOLO_WEIGHTS_PATH = ROOT / "YOLO" / "runs" / "segment" / "weights" / "best.pt"
    if not YOLO_WEIGHTS_PATH.exists():
        YOLO_WEIGHTS_PATH = ROOT / "YOLO" / "yolov8n-seg.pt"
        
    if YOLO_WEIGHTS_PATH.exists():
        print(f"Loading YOLO weights from {YOLO_WEIGHTS_PATH}...")
        yolo_model = YOLO(str(YOLO_WEIGHTS_PATH))
        yolo_model.to(device)
    else:
        print(f"WARNING: YOLO weights file not found. Pretrained weights will be downloaded automatically by ultralytics.")
        yolo_model = YOLO("yolov8n-seg.pt")
        yolo_model.to(device)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

# ---------------------------------------------------------------------------------------------
# Comparison-paper reproductions (comparison/ghorbani, comparison/yoon) - loaded read-only for
# the sidebar's MODELS radio (js/dom.js getSelectedModel), which lets the web UI show each
# paper's own labeling on the current image instead of CHAI's. Neither has an Arch Complexity
# model of its own (that's CHAI-specific geometry, not part of either paper), so their responses
# are just FDI-number boxes - the frontend renders those alone, no complexity/missing panel (see
# js/render.js redrawCanvas's non-CHAI branch).
# ---------------------------------------------------------------------------------------------
GHORBANI_DIR = ROOT / "comparison" / "ghorbani"
YOON_DIR = ROOT / "comparison" / "yoon"

ghorbani_module = None
ghorbani_detect_model = None
ghorbani_classify_model = None
ghorbani_eval_transform = None
yoon_model = None
yoon_inference_detector = None
# Class index -> FDI number for Yoon's detector output; Ghorbani's own FDI_CLASSES (identically
# ordered) comes from ghorbani_module directly. Defined unconditionally (cheap, no GPU/model
# involved) so /yoon_predict's own code can still reference it even when SKIP_COMPARISON_MODELS
# left yoon_model as None.
YOON_FDI_CLASSES = [q * 10 + d for q in (1, 2, 3, 4) for d in range(1, 7)]

if SKIP_COMPARISON_MODELS:
    print("SKIP_COMPARISON_MODELS=1 - /ghorbani_predict and /yoon_predict disabled, nothing "
          "extra loaded onto the GPU (useful when a training run already has it mostly full).")
else:
    try:
        sys.path.insert(0, str(GHORBANI_DIR))
        _ghorbani_spec = importlib.util.spec_from_file_location("ghorbani_evaluate", str(GHORBANI_DIR / "evaluate.py"))
        ghorbani_module = importlib.util.module_from_spec(_ghorbani_spec)
        _ghorbani_spec.loader.exec_module(ghorbani_module)

        ghorbani_detect_model = YOLO(str(ghorbani_module.find_latest_detect_weights()))

        classify_ckpt = GHORBANI_DIR / "model" / "classify_best.pth"
        if classify_ckpt.exists():
            ghorbani_classify_model = ghorbani_module.build_model(device)
            ghorbani_classify_model.load_state_dict(torch.load(classify_ckpt, map_location=device))
            ghorbani_classify_model.eval()
            ghorbani_eval_transform = transforms.Compose([
                transforms.Resize((ghorbani_module.IMG_SIZE, ghorbani_module.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            print("Ghorbani reproduction models loaded.")
        else:
            print(f"Ghorbani classify checkpoint not found at {classify_ckpt}; /ghorbani_predict disabled.")
    except Exception as e:
        print(f"Ghorbani reproduction models not available: {e}")

    try:
        from mmdet.apis import init_detector as _yoon_init_detector, inference_detector as _yoon_inference_detector
        sys.path.insert(0, str(YOON_DIR))
        _yoon_spec = importlib.util.spec_from_file_location("yoon_evaluate", str(YOON_DIR / "evaluate.py"))
        _yoon_evaluate_module = importlib.util.module_from_spec(_yoon_spec)
        _yoon_spec.loader.exec_module(_yoon_evaluate_module)

        yoon_checkpoint = _yoon_evaluate_module.find_checkpoint(None)
        yoon_model = _yoon_init_detector(str(YOON_DIR / "config.py"), yoon_checkpoint, device=str(device))
        yoon_inference_detector = _yoon_inference_detector
        print(f"Yoon reproduction model loaded from {yoon_checkpoint}.")
    except (Exception, SystemExit) as e:
        # find_checkpoint raises SystemExit (not Exception) when no checkpoint exists yet -
        # training may still be in progress, so this endpoint just stays disabled rather than
        # crashing startup.
        print(f"Yoon reproduction model not available: {e}")
        yoon_model = None


# Preprocessing transforms (must match ResNet/tooth/train.py validation transforms)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Shared with functions/graph/tooth.py's held-out evaluation (see that module's use of the same
# function, and ResNet/tooth/postprocess.py's docstring for why this replaced the old
# arch-refinement Transformer, now retired to backups/superseded_20260831/ViT_arch/).
_postprocess_spec = importlib.util.spec_from_file_location(
    "resnet_tooth_postprocess", str(ROOT / "ResNet" / "tooth" / "postprocess.py")
)
_postprocess_module = importlib.util.module_from_spec(_postprocess_spec)
_postprocess_spec.loader.exec_module(_postprocess_module)
resolve_arch_duplicates = _postprocess_module.resolve_arch_duplicates
correct_mirrors_by_digit_occurrence = _postprocess_module.correct_mirrors_by_digit_occurrence


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': yolo_model is not None and bool(tooth_models)})

@app.route('/segment', methods=['POST'])
def segment():
    if yolo_model is None:
        return jsonify({'success': False, 'error': 'YOLO model not loaded'}), 500
        
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty file uploaded'}), 400

    try:
        # Load image
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Run YOLO inference
        results = yolo_model(image, conf=0.25)
        
        predictions = []
        for r in results:
            boxes = r.boxes
            masks = r.masks
            if masks is None or len(masks) == 0:
                continue
                
            # Iterate through detections
            for i in range(len(masks)):
                # Get polygon coordinates
                poly = masks.xy[i].tolist() # List of [x, y] coordinates
                
                # Get box coordinates
                box = boxes.xyxy[i].cpu().numpy().tolist() # [x_min, y_min, x_max, y_max]
                
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = yolo_model.names[cls_id]
                
                predictions.append({
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': conf,
                    'box': box,
                    'polygon': poly
                })
                
        print(f"Segmented {len(predictions)} teeth contours.")
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/tooth_predict', methods=['POST'])
def tooth_predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({'success': False, 'error': 'Invalid JSON payload'}), 400

    jaw = payload.get('jaw')
    teeth = payload.get('teeth')

    if baseline_tooth_model is None:
        return jsonify({'success': False, 'error': 'Pooled tooth model not available'}), 400
    if jaw not in ('upper', 'lower'):
        return jsonify({'success': False, 'error': f'Invalid jaw: {jaw}'}), 400
    if not teeth or not isinstance(teeth, list):
        return jsonify({'success': False, 'error': 'No teeth provided for prediction'}), 400

    try:
        input_images = []
        input_meta = []

        for idx, tooth in enumerate(teeth):
            crop_data = tooth.get('crop')
            if not crop_data:
                return jsonify({'success': False, 'error': f'Crop image missing for tooth #{idx}'}), 400

            if ',' in crop_data:
                crop_data = crop_data.split(',', 1)[1]
            crop_bytes = base64.b64decode(crop_data)
            image = Image.open(io.BytesIO(crop_bytes)).convert('RGB')
            input_images.append(val_transform(image))

            # No theta - the pooled model's meta_fc is 4-dim (see coords_baseline.py /
            # train_baseline.py's ToothPositionClassifierNoTheta).
            meta_vector = [
                float(tooth.get('x1', 0.0)),
                float(tooth.get('y1', 0.0)),
                float(tooth.get('x2', 0.0)),
                float(tooth.get('y2', 0.0)),
            ]
            input_meta.append(torch.tensor(meta_vector, dtype=torch.float32))

        input_images = torch.stack(input_images).to(device)
        input_meta = torch.stack(input_meta).to(device)

        model = baseline_tooth_model
        with torch.no_grad():
            outputs = model(input_images, input_meta)

        probs = outputs.cpu().numpy()
        # Teeth arrive already ordered left-to-right along the arch (js/api.js
        # runToothAnalysis). Digits first, tens second - resolve the last digit arch-wide with NO
        # quadrant/tens information at all (resolve_arch_duplicates: capacity-2 Hungarian over
        # the whole arch, since a real arch has at most two teeth of any given digit regardless
        # of quadrant detection), then read the tens boundary off wherever a digit repeats
        # (correct_mirrors_by_digit_occurrence), falling back to the client's geometry-only
        # `mirror` guess only for a digit that appears just once. Doing it the other way -
        # grouping by the (occasionally wrong) geometric guess before resolving digits, like this
        # endpoint used to - let a bad quadrant call corrupt an already-correct neighboring
        # tooth's digit too; see ResNet/tooth/postprocess.py's docstrings for the full story.
        initial_mirrors = [bool(tooth.get('mirror', False)) for tooth in teeth]
        preds = resolve_arch_duplicates(probs)
        corrected_mirrors = correct_mirrors_by_digit_occurrence(preds.tolist(), initial_mirrors)
        confidences = probs[np.arange(len(preds)), preds]

        predictions = []
        for i in range(len(teeth)):
            predictions.append({
                'probs': [float(p) for p in probs[i].tolist()],
                'class_idx': int(preds[i]),
                'predicted_last_digit': int(preds[i]) + 1,
                'confidence': float(confidences[i]),
                'mirror': bool(corrected_mirrors[i])
            })

        return jsonify({'success': True, 'predictions': predictions})
    except Exception as e:
        print(f"Error during tooth prediction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/complexity_predict', methods=['POST'])
def complexity_predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({'success': False, 'error': 'Invalid JSON payload'}), 400

    jaw = payload.get('jaw')
    teeth = payload.get('teeth')

    if complexity_model is None:
        return jsonify({'success': False, 'error': 'Complexity model not available'}), 400
    if jaw not in ('lower', 'upper'):
        return jsonify({'success': False, 'error': f'Invalid jaw: {jaw}'}), 400
    if not teeth or not isinstance(teeth, list):
        return jsonify({'success': False, 'error': 'No teeth provided for prediction'}), 400

    try:
        # Teeth arrive already ordered left-to-right along the arch (js/api.js
        # runToothAnalysis), matching the order the model was trained on. Only the first
        # COMPLEXITY_MAX_TEETH get used - the model's positional embedding doesn't support
        # longer sequences.
        # No theta - COMPLEXITY_GEOM_DIM=4, matching the no-theta model loaded above (js/api.js's
        # payload already dropped the theta field to match).
        teeth = teeth[:COMPLEXITY_MAX_TEETH]
        geom = torch.tensor([
            [
                float(tooth.get('x1', 0.0)),
                float(tooth.get('y1', 0.0)),
                float(tooth.get('x2', 0.0)),
                float(tooth.get('y2', 0.0)),
            ]
            for tooth in teeth
        ], dtype=torch.float32, device=device).unsqueeze(0)
        key_padding_mask = torch.zeros(1, geom.size(1), dtype=torch.bool, device=device)

        with torch.no_grad():
            logits = complexity_model(geom, key_padding_mask)
            probs = torch.softmax(logits, dim=-1)[0]
            class_idx = int(torch.argmax(probs).item())

        return jsonify({
            'success': True,
            'class_idx': class_idx,
            'complexity': class_idx + 1,
            'probs': [float(p) for p in probs.cpu().tolist()]
        })
    except Exception as e:
        print(f"Error during complexity prediction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/ghorbani_predict', methods=['POST'])
def ghorbani_predict():
    if ghorbani_detect_model is None or ghorbani_classify_model is None:
        return jsonify({'success': False, 'error': 'Ghorbani reproduction model not available'}), 500
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file uploaded'}), 400

    file = request.files['image']
    try:
        pil_image = Image.open(io.BytesIO(file.read())).convert('RGB')
        # PIL gives RGB; the detector/crop_and_classify pair (comparison/ghorbani/evaluate.py)
        # was written and validated against cv2.imread's BGR convention - convert once here
        # rather than touching that already-working code.
        image = np.array(pil_image)[:, :, ::-1].copy()

        result = ghorbani_detect_model.predict(image, conf=ghorbani_module.CONF_THRESHOLD, verbose=False)[0]
        det_boxes = result.boxes.xyxy.cpu().numpy().tolist() if len(result.boxes) else []
        if not det_boxes:
            return jsonify({'success': True, 'predictions': [], 'jaw': None})

        det_probs = ghorbani_module.crop_and_classify(
            image, det_boxes, ghorbani_classify_model, device, ghorbani_eval_transform
        )

        # A live single-image request has no folder-derived jaw label the way evaluate.py's
        # offline test-set run does - infer it from a majority vote of the detections' own raw
        # top-1 quadrant, same spirit as resolve_duplicates_for_image's own per-half fallback.
        quadrant_of_class = np.array([c // 10 for c in ghorbani_module.FDI_CLASSES])
        top1_quadrant = quadrant_of_class[det_probs.argmax(axis=1)]
        upper_votes = int(np.isin(top1_quadrant, (1, 2)).sum())
        lower_votes = int(np.isin(top1_quadrant, (3, 4)).sum())
        jaw = 'upper' if upper_votes >= lower_votes else 'lower'

        final_class = ghorbani_module.resolve_duplicates_for_image(det_boxes, det_probs, jaw)

        predictions = []
        for box, cls, probs_row in zip(det_boxes, final_class, det_probs):
            cls = int(cls)
            predictions.append({
                'box': [float(v) for v in box],
                'fdi_number': cls,
                'confidence': float(probs_row[ghorbani_module.FDI_TO_IDX[cls]]),
            })
        return jsonify({'success': True, 'predictions': predictions, 'jaw': jaw})
    except Exception as e:
        print(f"Error during Ghorbani prediction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/yoon_predict', methods=['POST'])
def yoon_predict():
    if yoon_model is None:
        return jsonify({'success': False, 'error': 'Yoon reproduction model not available'}), 500
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file uploaded'}), 400

    file = request.files['image']
    try:
        pil_image = Image.open(io.BytesIO(file.read())).convert('RGB')
        # mmdet's own pipeline loads images BGR (mmcv's default, matching cv2/COCO convention) -
        # same conversion as /ghorbani_predict, for the same reason.
        image = np.array(pil_image)[:, :, ::-1].copy()

        result = yoon_inference_detector(yoon_model, image)
        pred = result.pred_instances
        keep = (pred.scores >= 0.25).cpu().numpy()
        det_boxes = pred.bboxes.cpu().numpy()[keep]
        det_scores = pred.scores.cpu().numpy()[keep]
        det_labels = pred.labels.cpu().numpy()[keep]

        # No duplicate-resolution post-process - Yoon et al.'s paper describes no such mechanism
        # for this model (unlike Ghorbani's Fig.1 half-jaw reassignment step), so its own top-1
        # class per detection is the faithful reproduction, not an omission - see
        # comparison/yoon/evaluate.py's module docstring for the full reasoning.
        predictions = []
        for box, score, label in zip(det_boxes, det_scores, det_labels):
            predictions.append({
                'box': [float(v) for v in box],
                'fdi_number': int(YOON_FDI_CLASSES[int(label)]),
                'confidence': float(score),
            })
        return jsonify({'success': True, 'predictions': predictions})
    except Exception as e:
        print(f"Error during Yoon prediction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # Start the server on port 5001. use_reloader explicitly off (not just relying on
    # debug=False's default) - Werkzeug's reloader was observed spawning a second process whose
    # sys.executable resolved to the system Python install instead of this project's .venv,
    # silently missing mmdet/mmengine there and leaving yoon_model None in the process actually
    # serving requests (server.py's own load-time print looked successful because it ran in the
    # OTHER process). All model loading above only needs to happen once.
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
