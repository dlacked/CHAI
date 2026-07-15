import io
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load config path if needed, or define locally for simplicity
ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "ResNet" / "jaw" / "jaw_classifier_model.pth"
if not WEIGHTS_PATH.exists():
    WEIGHTS_PATH = ROOT / "ResNet" / "jaw" / "best_classifier.pth"

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

# Define model structure (must match ResNet/jaw/train.py)
print("Initializing ResNet18 model...")
try:
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=None)
except (ImportError, AttributeError):
    from torchvision.models import resnet18
    model = resnet18()

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Load weights
if WEIGHTS_PATH.exists():
    print(f"Loading weights from {WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
else:
    print(f"WARNING: Weights file not found at {WEIGHTS_PATH}. Using untrained weights.")

model.to(device)
model.eval()

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

# Preprocessing transforms (must match ResNet/jaw/train.py validation transforms)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': WEIGHTS_PATH.exists()})

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty file uploaded'}), 400

    try:
        # Load image
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Apply transforms
        input_tensor = val_transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        classes = ["lower", "upper"]
        result = classes[predicted_idx.item()]
        score = confidence.item()
        
        print(f"Classified: {result} with confidence {score:.4f}")
        return jsonify({
            'success': True,
            'class': result,
            'confidence': score
        })
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

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

if __name__ == '__main__':
    # Start the server on port 5001
    app.run(host='0.0.0.0', port=5001, debug=False)
