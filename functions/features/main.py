import sys
import os
import csv
import argparse
import glob
import json
from pathlib import Path
from PIL import Image

# Ensure this directory is in sys.path for direct imports
features_dir = Path(__file__).resolve().parent
if str(features_dir) not in sys.path:
    sys.path.append(str(features_dir))

# Import our custom modules
from coords import get_normalized_coords
from fdi_last import get_centroid
from theta import get_theta_values, calculate_pca_rotation

def process_jaw(jaw, split, dataset_dir, output_csv):
    json_dir = Path(dataset_dir) / split / "labels_json" / jaw
    image_dir = Path(dataset_dir) / split / "images" / jaw
    
    if not json_dir.exists():
        print(f"Error: JSON labels directory {json_dir} does not exist.")
        return
        
    print(f"\n--- Starting GT feature extraction for {jaw.upper()} jaw ({split}) ---")
    print(f"JSON label directory: {json_dir}")
    
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    json_files = sorted(json_files)
    if not json_files:
        print(f"No JSON label files found in {json_dir}")
        return
        
    results_data = []
    
    for json_path in json_files:
        filename_no_ext = Path(json_path).stem
        
        # Find corresponding image to get sizes
        image_name = None
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp"]:
            test_img_path = image_dir / f"{filename_no_ext}{ext}"
            if test_img_path.exists():
                image_name = f"{filename_no_ext}{ext}"
                img_path = test_img_path
                break
                
        if image_name is None:
            print(f"Warning: Image file not found for {filename_no_ext}. Skipping.")
            continue
            
        try:
            # Get image size (width, height)
            with Image.open(img_path) as img:
                img_size = img.size  # (width, height)
                
            # Read ground-truth JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            teeth = data.get("tooth", [])
            if not teeth:
                continue
                
            segments_data = []
            for t in teeth:
                num = t.get("teeth_num")
                seg = t.get("segmentation", [])
                if not seg or num is None:
                    continue
                    
                # Format segmentation points
                if isinstance(seg[0], list) and len(seg[0]) == 2:
                    poly = seg
                else:
                    poly = [[seg[i], seg[i+1]] for i in range(0, len(seg), 2)]
                    
                # Centroid
                c = get_centroid(poly)

                segments_data.append({
                    "fdi_number": num,
                    "poly": poly,
                    "centroid": c
                })
                
            if not segments_data:
                continue
                
            # Calculate the PCA rotation (mean_pt, angle) via covariance eigen-decomposition -
            # the single PCA used everywhere: this CSV pipeline, the site's "PCA" overlay
            # checkbox, and computeToothMeta() at inference time (see js/geometry.js
            # calculatePCARotation, which mirrors this function exactly).
            centroids = [item["centroid"] for item in segments_data]
            mean_pt, angle = calculate_pca_rotation(centroids)

            # Get PCA rotated coordinates and theta values
            pca_results = get_theta_values(centroids, mean_pt, angle)
            
            # Add rotated_x and theta angle to segments_data
            for i, item in enumerate(segments_data):
                pca_info = pca_results[i]
                item["rotated_x"] = pca_info["rotated_x"]
                item["theta"] = pca_info["angle_norm"]  # Use the normalized angle
                
            # Sort left-to-right (by rotated_x)
            sorted_segments = sorted(segments_data, key=lambda x: x["rotated_x"])
            
            # Extract features for each sorted segment
            for seg_idx, seg_data in enumerate(sorted_segments):
                fdi_number = seg_data["fdi_number"]
                fdi_digit = fdi_number % 10

                # Rotate into the PCA frame, mirror (x flip), and normalize coordinates
                x1, y1, x2, y2 = get_normalized_coords(seg_data["poly"], mean_pt, angle, img_size, fdi_number)
                
                # Mirror theta angle if FDI tens digit is 2 or 3
                theta_val_raw = seg_data["theta"]
                if fdi_number != -1 and fdi_number is not None:
                    tens = fdi_number // 10
                    if tens in (2, 3):
                        if theta_val_raw >= 0:
                            theta_val_raw = 1.0 - theta_val_raw
                        else:
                            theta_val_raw = -1.0 - theta_val_raw
                
                theta_val = f"{theta_val_raw:.6f}"
                
                results_data.append([
                    image_name,
                    x1,
                    y1,
                    x2,
                    y2,
                    theta_val,
                    fdi_digit,
                    fdi_number
                ])
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing JSON {json_path}: {e}")
            continue
            
    if not results_data:
        print(f"No features extracted for {jaw.upper()} jaw.")
        return
        
    # Write to CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # fdi_number is the full 2-digit FDI tooth number - not a model input, it's only there so
    # ResNet/tooth/train.py's ToothDataset can look the tooth back up in the GT JSON for cropping
    headers = ["image_name", "x1", "y1", "x2", "y2", "theta", "fdi_last_digit", "fdi_number"]
    
    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(results_data)
        print(f"Success: Saved features to {output_csv.resolve()}")
    except Exception as e:
        print(f"Error writing CSV file {output_csv}: {e}")

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = project_root.parent / "dataset"
    output_dir = project_root / "ResNet" / "tooth" / "csv"
    
    parser = argparse.ArgumentParser(description="Extract combined coordinates, theta, and FDI features from GT JSONs.")
    parser.add_argument("--dataset_dir", type=str, default=str(dataset_dir),
                        help="Path to the dataset directory")
    parser.add_argument("--force", action="store_true",
                        help="Force feature extraction even if CSV files exist")
                        
    args = parser.parse_args()
    
    splits = ["train", "val"]
    jaws = ["lower", "upper"]
    
    for split in splits:
        for jaw in jaws:
            csv_path = output_dir / f"{jaw}_features_{split}.csv"
            if csv_path.exists() and not args.force:
                print(f"CSV file already exists: {csv_path.resolve()}. Skipping (use --force to overwrite).")
                continue
            process_jaw(jaw, split, args.dataset_dir, csv_path)
            
    print("\nFeature extraction completed successfully.")

if __name__ == "__main__":
    main()
