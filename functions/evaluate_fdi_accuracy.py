import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

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

def match_teeth_to_slots_clustered(sorted_teeth, W_pca, H_pca, mean_pt, cos_a, sin_a, is_lower):
    M = len(sorted_teeth)
    if M == 12:
        return list(range(12)), True
        
    # Calculate adjacent normalized distances using NumPy vectorization
    norm_dists = []
    for i in range(M - 1):
        polyA = sorted_teeth[i]["polygon"]
        polyB = sorted_teeth[i+1]["polygon"]
        
        arrA = np.array(polyA)  # (N, 2)
        arrB = np.array(polyB)  # (K, 2)
        
        # Calculate squared distances using broadcasting: shape (N, K)
        diff = arrA[:, np.newaxis, :] - arrB[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        ptA = polyA[min_idx[0]]
        ptB = polyB[min_idx[1]]
                    
        dx_rot = ptB[0] - ptA[0]
        dy_rot = ptB[1] - ptA[1]
        
        rxU = ptA[0] - mean_pt[0]
        ryU = ptA[1] - mean_pt[1]
        txU = rxU * cos_a - ryU * sin_a
        tyU = rxU * sin_a + ryU * cos_a

        rxV = ptB[0] - mean_pt[0]
        ryV = ptB[1] - mean_pt[1]
        txV = rxV * cos_a - ryV * sin_a
        tyV = rxV * sin_a + ryV * cos_a

        dx_rot = txV - txU
        dy_rot = tyV - tyU
        
        norm_dx = dx_rot / W_pca if W_pca > 0 else 0
        norm_dy = dy_rot / H_pca if H_pca > 0 else 0
        norm_weight = min(1.0, np.sqrt(norm_dx*norm_dx + norm_dy*norm_dy))
        norm_dists.append(norm_weight)
        
    S = 12 - M # Slots to skip
    
    # Find gap candidates where norm_dist > 0.05
    gap_candidates = []
    for i in range(M - 1):
        if norm_dists[i] > 0.05:
            gap_candidates.append({"index": i, "distance": norm_dists[i]})
            
    G = len(gap_candidates)
    if G == 0:
        return [0] * M, False  # No gaps detected -> cannot proceed
        
    # Sort by distance descending
    gap_candidates.sort(key=lambda x: x["distance"], reverse=True)
    
    gap_skips = {}
    if G >= S:
        # Top S gaps get 1 skip each
        for i in range(S):
            gap_skips[gap_candidates[i]["index"]] = 1
    else:
        # G < S: The largest gap gets S - G + 1 skips, others get 1 skip
        gap_skips[gap_candidates[0]["index"]] = S - G + 1
        for i in range(1, G):
            gap_skips[gap_candidates[i]["index"]] = 1
            
    mapped_slots = []
    current_slot = 0
    for i in range(M):
        mapped_slots.append(current_slot)
        skips = gap_skips.get(i, 0)
        current_slot += 1 + skips
        
    return [min(11, s) for s in mapped_slots], True

def evaluate():
    val_dir = r"c:\Users\chnyon\Desktop\lab\WORK\2026-1\CHAI\dataset\val\labels_json"
    
    # Accumulate results for Clustered Rule
    stats = {
        10: {"total": 0, "correct": 0, "total_teeth": 0, "correct_teeth": 0},
        11: {"total": 0, "correct": 0, "total_teeth": 0, "correct_teeth": 0},
        12: {"total": 0, "correct": 0, "total_teeth": 0, "correct_teeth": 0}
    }
    
    # Evaluate ONLY lower jaw
    for jaw_type in ["lower"]:
        json_dir = os.path.join(val_dir, jaw_type)
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        print(f"Scanning files in {json_dir}...")
        
        is_lower = (jaw_type == "lower")
        fdi_labels = [46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36] if is_lower else [16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26]
        
        for filepath in json_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            teeth = data.get("tooth", [])
            n = len(teeth)
            if n not in [10, 11, 12]:
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
                parsed_teeth.append({"gt_fdi": num, "centroid": c, "polygon": poly})
                
            if len(parsed_teeth) != n:
                continue
                
            # Perform PCA rotation to get rotated coordinates tx
            centroids = [t["centroid"] for t in parsed_teeth]
            mean_pt, angle = calculate_pca_rotation(centroids)
            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)
            
            for t in parsed_teeth:
                rx = t["centroid"][0] - mean_pt[0]
                ry = t["centroid"][1] - mean_pt[1]
                t["tx"] = rx * cos_a - ry * sin_a
                
            # Sort left-to-right (by tx)
            sorted_teeth = sorted(parsed_teeth, key=lambda x: x["tx"])
            
            # --- W_pca and H_pca calculation for normalization ---
            all_pts = []
            for t in parsed_teeth:
                all_pts.extend(t["polygon"])
                
            min_tx, max_tx = float('inf'), float('-inf')
            min_ty, max_ty = float('inf'), float('-inf')
            for pt in all_pts:
                rx = pt[0] - mean_pt[0]
                ry = pt[1] - mean_pt[1]
                tx = rx * cos_a - ry * sin_a
                ty = rx * sin_a + ry * cos_a
                if tx < min_tx: min_tx = tx
                if tx > max_tx: max_tx = tx
                if ty < min_ty: min_ty = ty
                if ty > max_ty: max_ty = ty
            W_pca = max_tx - min_tx if max_tx > min_tx else 600.0
            H_pca = max_ty - min_ty if max_ty > min_ty else 600.0
            
            # Map slots dynamically based on clustering
            slots, can_proceed = match_teeth_to_slots_clustered(sorted_teeth, W_pca, H_pca, mean_pt, cos_a, sin_a, is_lower)
                
            slice_correct = True
            if not can_proceed:
                slice_correct = False
                for tooth in sorted_teeth:
                    stats[n]["total_teeth"] += 1
            else:
                for idx, tooth in enumerate(sorted_teeth):
                    slot = slots[idx]
                    pred_fdi = fdi_labels[slot]
                    gt_fdi = tooth["gt_fdi"]
                    
                    stats[n]["total_teeth"] += 1
                    if pred_fdi == gt_fdi:
                        stats[n]["correct_teeth"] += 1
                    else:
                        slice_correct = False
                        
            stats[n]["total"] += 1
            if slice_correct:
                stats[n]["correct"] += 1
                
    print("\n=== Clustered (Gap-Based) Rule Evaluation Results (Lower Only) ===")
    for n in [10, 11, 12]:
        s = stats[n]
        if s["total"] > 0:
            slice_acc = s["correct"] / s["total"] * 100
            tooth_acc = s["correct_teeth"] / s["total_teeth"] * 100
            print(f"Teeth Count {n}:")
            print(f"  Slices: {s['correct']}/{s['total']} ({slice_acc:.2f}%)")
            print(f"  Teeth: {s['correct_teeth']}/{s['total_teeth']} ({tooth_acc:.2f}%)")
        else:
            print(f"Teeth Count {n}: No data")
            
    # Plot side-by-side subplots: Left (Slice-Level) and Right (Tooth-Level)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    counts = ["10 Teeth", "11 Teeth", "12 Teeth"]
    
    # 1. Left Subplot (Slice-Level)
    slice_accs = [
        (stats[10]["correct"] / stats[10]["total"] * 100) if stats[10]["total"] > 0 else 0,
        (stats[11]["correct"] / stats[11]["total"] * 100) if stats[11]["total"] > 0 else 0,
        (stats[12]["correct"] / stats[12]["total"] * 100) if stats[12]["total"] > 0 else 0
    ]
    bars1 = ax1.bar(counts, slice_accs, color=['#ff9999', '#66b3ff', '#99ff99'], edgecolor='grey', width=0.5)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Slice-Level Accuracy (%)", fontsize=12)
    ax1.set_title("Slice-Level Accuracy (Lower)", fontsize=13, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, height + 1, f"{height:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    # 2. Right Subplot (Tooth-Level)
    tooth_accs = [
        (stats[10]["correct_teeth"] / stats[10]["total_teeth"] * 100) if stats[10]["total_teeth"] > 0 else 0,
        (stats[11]["correct_teeth"] / stats[11]["total_teeth"] * 100) if stats[11]["total_teeth"] > 0 else 0,
        (stats[12]["correct_teeth"] / stats[12]["total_teeth"] * 100) if stats[12]["total_teeth"] > 0 else 0
    ]
    bars2 = ax2.bar(counts, tooth_accs, color=['#ffb3b3', '#80c0ff', '#b3ffb3'], edgecolor='grey', width=0.5)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Tooth-Level Accuracy (%)", fontsize=12)
    ax2.set_title("Tooth-Level Accuracy (Lower)", fontsize=13, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, height + 1, f"{height:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.suptitle("Clustered Gap-Based FDI Labeling Performance (Val Lower Only - Cleaned)", fontsize=15, fontweight='bold', y=0.98)
    
    # Save graph in functions/stats/graph directory
    output_dir = r"c:\Users\chnyon\Desktop\lab\WORK\2026-1\CHAI\CHAI\functions\stats\graph"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fdi_accuracy_by_count.png")

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nClustered Rule accuracy graph (Lower Only) saved successfully to: {output_path}")

if __name__ == "__main__":
    evaluate()
