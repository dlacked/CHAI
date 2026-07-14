import os
import json
import glob
import numpy as np

def calculate_pca_rotation(centroids):
    pts = np.array(centroids)
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    angle = np.arctan2(evecs[1, 0], evecs[0, 0])
    if angle > np.pi / 2:
        angle -= np.pi
    elif angle < -np.pi / 2:
        angle += np.pi
    return mean, angle

def get_centroid(segmentation):
    pts = np.array(segmentation)
    return np.mean(pts, axis=0).tolist()

def clean_dataset():
    dataset_root = r"c:\Users\chnyon\Desktop\lab\WORK\2026-1\CHAI\dataset"
    splits = ["train", "val"]
    jaws = ["lower", "upper"]
    
    total_deleted = 0
    
    for split in splits:
        for jaw in jaws:
            json_dir = os.path.join(dataset_root, split, "labels_json", jaw)
            if not os.path.exists(json_dir):
                print(f"Directory not found: {json_dir}")
                continue
                
            json_files = glob.glob(os.path.join(json_dir, "*.json"))
            print(f"\nScanning {len(json_files)} files in {split}/{jaw}...")
            
            is_lower = (jaw == "lower")
            
            for filepath in json_files:
                sample_name = os.path.splitext(os.path.basename(filepath))[0]
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                teeth = data.get("tooth", [])
                if not teeth:
                    continue
                    
                parsed_teeth = []
                for t in teeth:
                    num = t.get("teeth_num")
                    seg = t.get("segmentation", [])
                    if not seg or num is None:
                        continue
                    if isinstance(seg[0], list) and len(seg[0]) == 2:
                        poly = seg
                    else:
                        poly = [[seg[i], seg[i+1]] for i in range(0, len(seg), 2)]
                    
                    c = get_centroid(poly)
                    parsed_teeth.append({"gt_fdi": num, "centroid": c})
                    
                if not parsed_teeth:
                    continue
                    
                # Sort left-to-right
                if len(parsed_teeth) > 1:
                    centroids = [t["centroid"] for t in parsed_teeth]
                    try:
                        mean_pt, angle = calculate_pca_rotation(centroids)
                        cos_a = np.cos(-angle)
                        sin_a = np.sin(-angle)
                        for t in parsed_teeth:
                            rx = t["centroid"][0] - mean_pt[0]
                            ry = t["centroid"][1] - mean_pt[1]
                            t["tx"] = rx * cos_a - ry * sin_a
                        sorted_teeth = sorted(parsed_teeth, key=lambda x: x["tx"])
                    except Exception:
                        sorted_teeth = sorted(parsed_teeth, key=lambda x: x["centroid"][0])
                else:
                    sorted_teeth = parsed_teeth
                    
                leftmost_fdi = sorted_teeth[0]["gt_fdi"]
                leftmost_digit = leftmost_fdi // 10
                
                is_mirrored = False
                if is_lower:
                    is_mirrored = (leftmost_digit == 3)
                else:
                    is_mirrored = (leftmost_digit == 2)
                    
                if is_mirrored:
                    print(f"[{split}/{jaw}] Mirrored data detected: {sample_name} (Leftmost FDI: {leftmost_fdi}) - Deleting...")
                    
                    # 1. Delete JSON file
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        print(f"Error removing JSON: {e}")
                        
                    # 2. Delete YOLO label (.txt)
                    txt_path = os.path.join(dataset_root, split, "labels", jaw, f"{sample_name}.txt")
                    if os.path.exists(txt_path):
                        try:
                            os.remove(txt_path)
                        except Exception as e:
                            print(f"Error removing TXT label: {e}")
                            
                    # 3. Delete matching images
                    img_pattern = os.path.join(dataset_root, split, "images", jaw, f"{sample_name}.*")
                    for img_path in glob.glob(img_pattern):
                        try:
                            os.remove(img_path)
                        except Exception as e:
                            print(f"Error removing image: {e}")
                            
                    # 4. Delete matching masking images
                    mask_pattern = os.path.join(dataset_root, split, "masking_images", jaw, f"{sample_name}.*")
                    for mask_path in glob.glob(mask_pattern):
                        try:
                            os.remove(mask_path)
                        except Exception as e:
                            print(f"Error removing masking image: {e}")
                            
                    total_deleted += 1
                    
    print(f"\nCleanup complete. Total mirrored samples deleted: {total_deleted}")

if __name__ == "__main__":
    clean_dataset()
