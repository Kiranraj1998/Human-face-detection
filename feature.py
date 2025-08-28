# step3_feature_engineering.py

import numpy as np
import pandas as pd
import cv2
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray

# ---- 1. LOAD CLEANED ANNOTATIONS & PREPROCESSED IMAGES ----
df = pd.read_csv('faces_cleaned.csv')  
data = np.load('faces_data_compressed.npz')
faces = data['clean']   # shape: (N, 224, 224, 3)

# ---- 2. BOUNDING-BOX COORDINATE FEATURES ----
# Normalize original bbox coords by original image size
bbox_feats = []
for _, row in df.iterrows():
    w0, h0 = row['width'], row['height']
    x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
    cx = ((x0 + x1) / 2) / w0
    cy = ((y0 + y1) / 2) / h0
    bw = (x1 - x0) / w0
    bh = (y1 - y0) / h0
    bbox_feats.append([cx, cy, bw, bh])
bbox_feats = np.array(bbox_feats)  # shape: (N, 4)

# ---- 3. IMAGE-BASED FEATURES (Equalization → HOG → LBP) ----
def extract_img_feats(img):
    # 3.1 to uint8 grayscale & equalize
    gray = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_BGR2GRAY)
    eq   = cv2.equalizeHist(gray)
    eq_f = eq / 255.0

    # 3.2 HOG
    hog_feat = hog(
        eq_f,
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        feature_vector=True
    )

    # 3.3 LBP + normalized histogram
    lbp = local_binary_pattern(
        eq,
        P=8, R=1,
        method='uniform'
    )
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )

    return np.concatenate([hog_feat, lbp_hist])

# Compute for all faces
print("🔍 Extracting image features (HOG + LBP)...")
img_feats = np.array([extract_img_feats(f) for f in faces])
print("   Image feature matrix shape:", img_feats.shape)

# ---- 4. COMBINE FEATURES ----
X = np.hstack([bbox_feats, img_feats])
print("✅ Combined feature matrix shape:", X.shape)

# ---- 5. SAVE FEATURES ----
np.save('features.npy', X)
print("➡️ Saved features.npy for downstream modeling")
