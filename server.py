import io
import base64
import importlib.util
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load config path if needed, or define locally for simplicity
ROOT = Path(__file__).resolve().parent

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
# same dynamic-import approach used for the ViT models below) instead of keeping a
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

# Load the arch-level Transformer that refines each tooth's ResNet prediction using self-attention
# over the whole dental arch (see ViT/arch/model.py). Loaded by file path rather than sys.path +
# import, so this doesn't depend on ViT/arch being an importable package or collide with any
# other "model" module name.
ARCH_MODEL_PATH = ROOT / "ViT" / "arch" / "model.py"
ARCH_MODEL_DIR = ROOT / "ViT" / "arch" / "model"
ARCH_MAX_TEETH = 12  # matches ViT/arch/build_dataset.py's MAX_TEETH_PER_ARCH / the model's pos_embed length

arch_models = {}
if ARCH_MODEL_PATH.exists():
    spec = importlib.util.spec_from_file_location("vit_arch_model", str(ARCH_MODEL_PATH))
    _vit_arch_model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_vit_arch_model_module)
    ArchToothTransformer = _vit_arch_model_module.ArchToothTransformer

    for jaw_name in ("lower", "upper"):
        weights_path = ARCH_MODEL_DIR / f"{jaw_name}_best.pth"
        if weights_path.exists():
            print(f"Loading arch transformer for {jaw_name} from {weights_path}...")
            arch_model = ArchToothTransformer(num_classes=6)
            arch_model.load_state_dict(torch.load(weights_path, map_location=device))
            arch_model.to(device)
            arch_model.eval()
            arch_models[jaw_name] = arch_model
        else:
            print(f"Arch transformer weights not found for {jaw_name}: {weights_path} (falling back to ResNet-only for this jaw)")
else:
    print(f"ViT/arch/model.py not found at {ARCH_MODEL_PATH}; arch-level refinement disabled.")

# Load the arch-level Transformer that predicts an arch's overall complexity class (I/II/III)
# from per-tooth geometry alone - no image or ResNet features involved (see
# ViT/complexity/model.py). Same dynamic-import approach as the ArchToothTransformer above.
COMPLEXITY_MODEL_PATH = ROOT / "ViT" / "complexity" / "model.py"
COMPLEXITY_MODEL_DIR = ROOT / "ViT" / "complexity" / "model"
COMPLEXITY_MAX_TEETH = 12  # matches ViT/complexity/build_dataset.py's MAX_TEETH_PER_ARCH

complexity_models = {}
if COMPLEXITY_MODEL_PATH.exists():
    spec = importlib.util.spec_from_file_location("vit_complexity_model", str(COMPLEXITY_MODEL_PATH))
    _vit_complexity_model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_vit_complexity_model_module)
    ArchComplexityTransformer = _vit_complexity_model_module.ArchComplexityTransformer

    for jaw_name in ("lower", "upper"):
        weights_path = COMPLEXITY_MODEL_DIR / f"{jaw_name}_best.pth"
        if weights_path.exists():
            print(f"Loading complexity transformer for {jaw_name} from {weights_path}...")
            complexity_model = ArchComplexityTransformer(num_classes=3)
            complexity_model.load_state_dict(torch.load(weights_path, map_location=device))
            complexity_model.to(device)
            complexity_model.eval()
            complexity_models[jaw_name] = complexity_model
        else:
            print(f"Complexity transformer weights not found for {jaw_name}: {weights_path}")
else:
    print(f"ViT/complexity/model.py not found at {COMPLEXITY_MODEL_PATH}; complexity classification disabled.")

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

# Preprocessing transforms (must match ResNet/tooth/train.py validation transforms)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

    if jaw not in tooth_models:
        return jsonify({'success': False, 'error': f'Tooth model not available for jaw: {jaw}'}), 400
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

            meta_vector = [
                float(tooth.get('x1', 0.0)),
                float(tooth.get('y1', 0.0)),
                float(tooth.get('x2', 0.0)),
                float(tooth.get('y2', 0.0)),
                float(tooth.get('theta', 0.0))
            ]
            input_meta.append(torch.tensor(meta_vector, dtype=torch.float32))

        input_images = torch.stack(input_images).to(device)
        input_meta = torch.stack(input_meta).to(device)

        model = tooth_models[jaw]
        arch_model = arch_models.get(jaw)
        refined = False
        with torch.no_grad():
            # Run the ResNet backbone in stages (instead of model(img, meta), which only
            # returns the final softmax) so the image embedding is available to feed the arch
            # transformer - mirrors ViT/arch/build_dataset.py's cache-building forward pass.
            img_features = model.resnet(input_images)
            meta_features = model.meta_fc(input_meta)
            logits = model.classifier(torch.cat((img_features, meta_features), dim=1))
            outputs = torch.softmax(logits, dim=1)

            if arch_model is not None:
                # Teeth arrive already ordered left-to-right along the arch (js/api.js
                # runToothAnalysis), matching the order the arch transformer was trained on.
                # Only the first ARCH_MAX_TEETH get refined - the model's positional embedding
                # doesn't support longer sequences (over-detection past 12 teeth is already a
                # rare edge case in this pipeline).
                seq_len = min(img_features.size(0), ARCH_MAX_TEETH)
                geom = input_meta[:seq_len].unsqueeze(0)
                img_vec = img_features[:seq_len].unsqueeze(0)
                prob_vec = outputs[:seq_len].unsqueeze(0)
                key_padding_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)

                refined_logits = arch_model(geom, img_vec, prob_vec, key_padding_mask)
                refined_probs = torch.softmax(refined_logits, dim=-1)[0]

                if seq_len < outputs.size(0):
                    outputs = outputs.clone()
                    outputs[:seq_len] = refined_probs
                else:
                    outputs = refined_probs
                refined = True

            confidences, preds = torch.max(outputs, dim=1)

        predictions = []
        for i in range(len(teeth)):
            predictions.append({
                'probs': [float(p) for p in outputs[i].cpu().tolist()],
                'class_idx': int(preds[i].item()),
                'predicted_last_digit': int(preds[i].item()) + 1,
                'confidence': float(confidences[i].item())
            })

        return jsonify({'success': True, 'predictions': predictions, 'refined': refined})
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

    if jaw not in complexity_models:
        return jsonify({'success': False, 'error': f'Complexity model not available for jaw: {jaw}'}), 400
    if not teeth or not isinstance(teeth, list):
        return jsonify({'success': False, 'error': 'No teeth provided for prediction'}), 400

    try:
        # Teeth arrive already ordered left-to-right along the arch (js/api.js
        # runToothAnalysis), matching the order the model was trained on. Only the first
        # COMPLEXITY_MAX_TEETH get used - the model's positional embedding doesn't support
        # longer sequences.
        teeth = teeth[:COMPLEXITY_MAX_TEETH]
        geom = torch.tensor([
            [
                float(tooth.get('x1', 0.0)),
                float(tooth.get('y1', 0.0)),
                float(tooth.get('x2', 0.0)),
                float(tooth.get('y2', 0.0)),
                float(tooth.get('theta', 0.0)),
            ]
            for tooth in teeth
        ], dtype=torch.float32, device=device).unsqueeze(0)
        key_padding_mask = torch.zeros(1, geom.size(1), dtype=torch.bool, device=device)

        model = complexity_models[jaw]
        with torch.no_grad():
            logits = model(geom, key_padding_mask)
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

if __name__ == '__main__':
    # Start the server on port 5001
    app.run(host='0.0.0.0', port=5001, debug=False)
