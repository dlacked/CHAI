import os
import csv
import argparse
import glob
import numpy as np
from pathlib import Path
from PIL import Image

def calculate_pca_rotation(centroids):
    """
    Calculate the PCA rotation angle and mean point (center) for a list of centroids.
    """
    pts = np.array(centroids)
    if len(pts) < 2:
        return np.mean(pts, axis=0) if len(pts) == 1 else np.zeros(2), 0.0
    
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    
    if cov.ndim == 0 or np.allclose(cov, 0):
        return mean, 0.0
        
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    angle = np.arctan2(evecs[1, 0], evecs[0, 0])
    
    if angle > np.pi / 2:
        angle -= np.pi
    elif angle < -np.pi / 2:
        angle += np.pi
    return mean, angle

def get_centroid(polygon_points):
    """
    Calculate the centroid of a polygon (list of [x, y] coordinates).
    """
    pts = np.array(polygon_points)
    return np.mean(pts, axis=0).tolist()

def extract_coords(input_dir, output_csv, yolo_weights, conf):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: 'ultralytics' package is not installed. Please install it to run YOLO segmentation.")
        return

    print(f"Loading YOLO segmentation model from: {yolo_weights}")
    if not os.path.exists(yolo_weights):
        print(f"Warning: YOLO weights file not found at {yolo_weights}. Fallback to online yolov8n-seg.pt")
        model = YOLO("yolov8n-seg.pt")
    else:
        model = YOLO(yolo_weights)
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(str(input_path / ext)))
        
    image_files = sorted(list(set(image_files)))
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images. Starting coordinates (bounding box) extraction...")

    results_data = []

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"Processing: {filename}")
        
        try:
            image = Image.open(img_path).convert('RGB')
            inference_results = model(image, conf=conf, verbose=False)
            
            # Extract polygons and boxes in pairs
            polygons_and_boxes = []
            for r in inference_results:
                masks = r.masks
                boxes = r.boxes
                if masks is None or len(masks) == 0:
                    continue
                for i in range(len(masks)):
                    poly = masks.xy[i].tolist()
                    if len(poly) > 0:
                        # xyxy is [x1, y1, x2, y2]
                        box = boxes.xyxy[i].cpu().numpy().tolist()
                        polygons_and_boxes.append((poly, box))
            
            if not polygons_and_boxes:
                print(f"  -> No segments detected in {filename}. Skipping.")
                continue
                
            # Calculate centroids for PCA sorting
            centroids = [get_centroid(item[0]) for item in polygons_and_boxes]
            mean_pt, angle = calculate_pca_rotation(centroids)
            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)
            
            # Project each segment to PCA coordinate to sort left-to-right
            segments_info = []
            for idx, (poly, box) in enumerate(polygons_and_boxes):
                c = centroids[idx]
                rx = c[0] - mean_pt[0]
                ry = c[1] - mean_pt[1]
                
                # Rotated coordinates (tx, ty) for sorting
                tx = rx * cos_a - ry * sin_a
                
                segments_info.append({
                    "rotated_x": tx,
                    "box": box  # [x1, y1, x2, y2]
                })
            
            # Sort left-to-right based on rotated_x (tx)
            segments_info = sorted(segments_info, key=lambda x: x["rotated_x"])
            
            # Append sorted coordinates
            for seg_idx, info in enumerate(segments_info):
                box = info["box"]
                results_data.append([
                    filename,
                    seg_idx,
                    f"{box[0]:.4f}",
                    f"{box[1]:.4f}",
                    f"{box[2]:.4f}",
                    f"{box[3]:.4f}"
                ])
                
        except Exception as e:
            print(f"  -> Error processing {filename}: {str(e)}")
            continue

    if not results_data:
        print("No coordinates extracted. CSV file will not be created.")
        return

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        "image_name", 
        "segment_idx", 
        "x1", 
        "y1", 
        "x2", 
        "y2"
    ]
    
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(results_data)
        print(f"\nCoordinates extraction complete! Results saved to: {output_path.resolve()}")
    except Exception as e:
        print(f"Error writing CSV file: {str(e)}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    default_weights = project_root / "YOLO" / "runs" / "segment" / "weights" / "best.pt"
    if not default_weights.exists():
        default_weights = project_root / "YOLO" / "yolov8n-seg.pt"

    parser = argparse.ArgumentParser(description="Extract bounding box coordinates of tooth segments using YOLO + PCA.")
    parser.add_argument("--input_dir", type=str, default="resnet_features/train",
                        help="Path to directory containing input training images (default: resnet_features/train)")
    parser.add_argument("--output_csv", type=str, default="coords.csv",
                        help="Path to output CSV file (default: coords.csv)")
    parser.add_argument("--yolo_weights", type=str, default=str(default_weights),
                        help="Path to YOLO segmentation weights (default: YOLO best.pt or yolov8n-seg.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for YOLO segmentation (default: 0.25)")
    
    args = parser.parse_args()
    
    extract_coords(args.input_dir, args.output_csv, args.yolo_weights, args.conf)
