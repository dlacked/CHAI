import os
import glob
import numpy as np

def calculate_pca_rotation(polygon_points):
    pts = np.array(polygon_points)
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

def get_centroid(polygon):
    pts = np.array(polygon)
    return np.mean(pts, axis=0)

def process_labels_dir(labels_dir):
    all_gaps = []
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    print(f"Scanning {len(txt_files)} files in {labels_dir}...")
    
    for filepath in txt_files:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        if len(lines) != 12:
            continue
            
        polygons = []
        all_pts = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            coords = [float(x) for x in parts[1:]]
            poly = [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]
            polygons.append(poly)
            all_pts.extend(poly)
            
        if len(polygons) != 12:
            continue
            
        mean_pt, angle = calculate_pca_rotation(all_pts)
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        
        rotated_pts = []
        for pt in all_pts:
            rx = pt[0] - mean_pt[0]
            ry = pt[1] - mean_pt[1]
            tx = rx * cos_a - ry * sin_a
            rotated_pts.append(tx)
            
        min_tx = min(rotated_pts)
        max_tx = max(rotated_pts)
        W_pca = max_tx - min_tx
        if W_pca <= 0:
            continue
            
        centroids = [get_centroid(poly) for poly in polygons]
        rotated_centroids = []
        for c in centroids:
            rx = c[0] - mean_pt[0]
            ry = c[1] - mean_pt[1]
            tx = rx * cos_a - ry * sin_a
            rotated_centroids.append(tx)
            
        sorted_tx = sorted(rotated_centroids)
        
        gaps = []
        for i in range(11):
            gap = sorted_tx[i+1] - sorted_tx[i]
            gaps.append(gap / W_pca)
            
        all_gaps.append(gaps)
        
    if not all_gaps:
        return None
        
    all_gaps = np.array(all_gaps)
    mean_gaps = np.mean(all_gaps, axis=0)
    return mean_gaps.tolist()

if __name__ == "__main__":
    train_dir = r"c:\Users\chnyon\Desktop\lab\WORK\2026-1\CHAI\dataset\train\labels"
    
    print("Processing Lower Jaw (Mandible)...")
    lower_ratios = process_labels_dir(os.path.join(train_dir, "lower"))
    print("Lower Ratios:", lower_ratios)
    
    print("\nProcessing Upper Jaw (Maxilla)...")
    upper_ratios = process_labels_dir(os.path.join(train_dir, "upper"))
    print("Upper Ratios:", upper_ratios)
