# convert_csv_to_yolo_labels.py

import os
import pandas as pd
from pathlib import Path

# ---- SETTINGS ----
CSV_PATH  = 'faces_cleaned.csv'   # your cleaned annotation CSV
IMG_DIR   = 'images/images'       # folder containing .jpg images
LABEL_DIR = 'labels/images'       # where YOLO expects .txt labels
CLASS_ID  = 0                     # zero‐based class index for “face”

# 1. Read the cleaned annotations
df = pd.read_csv(CSV_PATH)

# 2. Make sure the target directory exists
os.makedirs(LABEL_DIR, exist_ok=True)

# 3. Group by image and write one .txt per image
for image_name, group in df.groupby('image_name'):
    # original image dimensions (from your CSV)
    w = group.iloc[0]['width']
    h = group.iloc[0]['height']
    
    # build each line: CLASS_ID x_center_norm y_center_norm width_norm height_norm
    lines = []
    for _, row in group.iterrows():
        x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
        cx = ((x0 + x1) / 2) / w
        cy = ((y0 + y1) / 2) / h
        bw = (x1 - x0) / w
        bh = (y1 - y0) / h
        lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # write to labels/images/<stem>.txt
    stem = Path(image_name).stem
    label_path = Path(LABEL_DIR) / f"{stem}.txt"
    with open(label_path, 'w') as f:
        f.write("\n".join(lines))

print(f"✅ Wrote {len(df['image_name'].unique())} label files to {LABEL_DIR}/")
