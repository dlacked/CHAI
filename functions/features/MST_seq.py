import os
import csv
import json
import argparse
from pathlib import Path
import numpy as np

def get_mst_seq(fdi_number):
    """
    Calculate the MST sequence index (0-based) from the full FDI tooth number.
    - base = ones_digit - 1
    - if tens_digit in (2, 3): value = base + 6
    - else: value = base
    """
    if fdi_number == -1 or fdi_number is None:
        return -1
    
    tens = fdi_number // 10
    ones = fdi_number % 10
    if tens in (1, 4):
        return int(6 - ones)
    elif tens in (2, 3):
        return int(5 + ones)
    return -1

def update_csv_file(csv_path, dataset_dir, jaw):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    print(f"Updating {csv_path} with mst_seq column...")
    
    from fdi_last import get_fdi_number
    
    rows = []
    headers = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            rows.append(row)
            
    mst_seq_idx = -1
    if "mst_seq" in headers:
        mst_seq_idx = headers.index("mst_seq")
    else:
        headers.append("mst_seq")
        
    updated_rows = []
    for row in rows:
        image_name = row[0]
        try:
            x1 = float(row[2])
            y1 = float(row[3])
            x2 = float(row[4])
            y2 = float(row[5])
            
            poly_mock = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            fdi_number = get_fdi_number(image_name, poly_mock, dataset_dir, jaw)
            mst_val = get_mst_seq(fdi_number)
        except Exception as e:
            print(f"Error calculating mst_seq for row {row}: {e}")
            mst_val = -1
            
        if mst_seq_idx != -1:
            row[mst_seq_idx] = str(mst_val)
        else:
            row.append(str(mst_val))
        updated_rows.append(row)
        
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(updated_rows)
        
    print(f"Successfully updated {csv_path}")

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = project_root.parent / "dataset"
    output_dir = project_root / "ResNet" / "tooth" / "csv"
    
    lower_csv = output_dir / "lower_features.csv"
    upper_csv = output_dir / "upper_features.csv"
    
    if lower_csv.exists():
        update_csv_file(lower_csv, dataset_dir, "lower")
    if upper_csv.exists():
        update_csv_file(upper_csv, dataset_dir, "upper")

if __name__ == "__main__":
    main()
