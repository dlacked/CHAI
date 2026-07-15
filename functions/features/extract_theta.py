import sys
import os
import csv
import argparse
import numpy as np
from pathlib import Path

# Add current directory to path
features_dir = Path(__file__).resolve().parent
if str(features_dir) not in sys.path:
    sys.path.append(str(features_dir))

from segmentation import run_segmentation
from theta import get_theta_values
from fdi_last import get_centroid

def extract_theta(input_dir, output_csv, yolo_weights, conf):
    seg_results = run_segmentation(input_dir, yolo_weights, conf)
    if not seg_results:
        print("No segments detected.")
        return

    results_data = []
    for filename, segments in seg_results.items():
        if not segments:
            continue
        try:
            centroids = [get_centroid(item[0]) for item in segments]
            pca_results = get_theta_values(centroids)
            
            segments_info = []
            for i, (poly, box) in enumerate(segments):
                pca_info = pca_results[i]
                segments_info.append({
                    "centroid_x": centroids[i][0],
                    "centroid_y": centroids[i][1],
                    "rotated_x": pca_info["rotated_x"],
                    "rotated_y": pca_info["rotated_y"],
                    "angle_rad": pca_info["angle_rad"],
                    "angle_deg": np.degrees(pca_info["angle_rad"])
                })
                
            sorted_segments = sorted(segments_info, key=lambda x: x["rotated_x"])
            
            for seg_idx, info in enumerate(sorted_segments):
                results_data.append([
                    filename,
                    seg_idx,
                    f"{info['centroid_x']:.4f}",
                    f"{info['centroid_y']:.4f}",
                    f"{info['rotated_x']:.4f}",
                    f"{info['rotated_y']:.4f}",
                    f"{info['angle_rad']:.6f}",
                    f"{info['angle_deg']:.4f}"
                ])
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    if not results_data:
        print("No theta values extracted.")
        return

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        "image_name", 
        "segment_idx", 
        "centroid_x", 
        "centroid_y", 
        "rotated_x", 
        "rotated_y", 
        "angle_rad", 
        "angle_deg"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results_data)
    print(f"Theta values saved to {output_path.resolve()}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    default_weights = project_root / "YOLO" / "runs" / "segment" / "weights" / "best.pt"
    if not default_weights.exists():
        default_weights = project_root / "YOLO" / "yolov8n-seg.pt"

    parser = argparse.ArgumentParser(description="Extract angles (theta) of tooth segments using YOLO + PCA.")
    parser.add_argument("--input_dir", type=str, default="resnet_features/train",
                        help="Path to directory containing input training images")
    parser.add_argument("--output_csv", type=str, default="theta.csv",
                        help="Path to output CSV file")
    parser.add_argument("--yolo_weights", type=str, default=str(default_weights),
                        help="Path to YOLO segmentation weights")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for YOLO segmentation")
    
    args = parser.parse_args()
    extract_theta(args.input_dir, args.output_csv, args.yolo_weights, args.conf)
