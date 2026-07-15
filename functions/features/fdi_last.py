import os
import json
import numpy as np
from pathlib import Path

def get_centroid(polygon_points):
    """
    Calculate the centroid of a polygon.
    """
    pts = np.array(polygon_points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        pts = pts.reshape(-1, 2)
    return np.mean(pts, axis=0).tolist()

def get_fdi_number(image_name, yolo_polygon, dataset_dir, jaw):
    """
    Matches the YOLO segment centroid with the ground truth JSON centroids 
    and returns the full matched FDI tooth number.
    Args:
        image_name (str): Name of the image file (e.g. lower_1.png).
        yolo_polygon (list): List of [x, y] coordinates of the YOLO mask.
        dataset_dir (Path or str): Path to the dataset folder.
        jaw (str): "lower" or "upper".
    Returns:
        int: The matched FDI tooth number (e.g., 46, 31), or -1 if no match found.
    """
    image_name_no_ext = os.path.splitext(image_name)[0]
    json_path = Path(dataset_dir) / "train" / "labels_json" / jaw / f"{image_name_no_ext}.json"
    
    if not json_path.exists():
        return -1
        
    try:
        yolo_c = np.array(get_centroid(yolo_polygon))
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        best_fdi = -1
        min_dist = float('inf')
        
        for t in data.get("tooth", []):
            num = t.get("teeth_num")
            seg = t.get("segmentation", [])
            if not seg or num is None:
                continue
            
            if isinstance(seg[0], list) and len(seg[0]) == 2:
                poly = seg
            else:
                poly = [[seg[i], seg[i+1]] for i in range(0, len(seg), 2)]
                
            gt_c = np.array(get_centroid(poly))
            
            dist = np.linalg.norm(yolo_c - gt_c)
            if dist < min_dist:
                min_dist = dist
                best_fdi = num
                
        # Accept matching if centroids are reasonably close
        if min_dist < 200 and best_fdi != -1:
            return int(best_fdi)
        return -1
    except Exception as e:
        print(f"Error matching FDI for {image_name}: {e}")
        return -1

def get_fdi_last_digit(image_name, yolo_polygon, dataset_dir, jaw):
    """
    Matches the YOLO segment centroid with the ground truth JSON centroids 
    and returns the last digit of the matched FDI tooth number.
    """
    fdi_number = get_fdi_number(image_name, yolo_polygon, dataset_dir, jaw)
    if fdi_number != -1:
        return fdi_number % 10
    return -1
