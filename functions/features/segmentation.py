import os
import glob
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

def run_segmentation(input_dir, yolo_weights=None, conf=0.25):
    """
    Runs YOLO segmentation on images in input_dir and returns predicted segments.
    Args:
        input_dir (str or Path): Path to directory containing images.
        yolo_weights (str or Path, optional): Path to YOLO weights file.
        conf (float): Confidence threshold for YOLO inference.
    Returns:
        dict: {filename: [(polygon, box), ...]}
        Where polygon is a list of [x, y] coordinates, and box is [x1, y1, x2, y2] list.
    """
    if yolo_weights is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        yolo_weights = project_root / "YOLO" / "runs" / "segment" / "weights" / "best.pt"
        if not yolo_weights.exists():
            yolo_weights = project_root / "YOLO" / "yolov8n-seg.pt"
            
    yolo_weights = str(yolo_weights)
    print(f"Loading YOLO segmentation model from: {yolo_weights}")
    
    if not os.path.exists(yolo_weights):
        print(f"Warning: YOLO weights file not found. Fallback to yolov8n-seg.pt")
        model = YOLO("yolov8n-seg.pt")
    else:
        model = YOLO(yolo_weights)
        
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return {}

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(str(input_path / ext)))
        
    image_files = sorted(list(set(image_files)))
    if not image_files:
        print(f"No images found in {input_dir}")
        return {}

    results = {}
    for img_path in image_files:
        filename = os.path.basename(img_path)
        try:
            image = Image.open(img_path).convert('RGB')
            # Run inference
            inference_results = model(image, conf=conf, verbose=False)
            
            polygons_and_boxes = []
            for r in inference_results:
                masks = r.masks
                boxes = r.boxes
                if masks is None or len(masks) == 0:
                    continue
                for i in range(len(masks)):
                    poly = masks.xy[i].tolist()
                    if len(poly) > 0:
                        box = boxes.xyxy[i].cpu().numpy().tolist()
                        polygons_and_boxes.append((poly, box))
                        
            results[filename] = polygons_and_boxes
        except Exception as e:
            print(f"Error segmenting {filename}: {e}")
            
    return results
