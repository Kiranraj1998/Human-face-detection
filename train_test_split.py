import numpy as np
from sklearn.model_selection import train_test_split

# ---- 1. LOAD FEATURES & LABELS ----
# You should already have your feature matrix from Step 3:
X = np.load('features.npy')         # shape: (N, D)

# If you have a label array (e.g. bounding‐box targets or class labels), load it here:
# y = np.load('labels.npy')         # shape: (N,)
# If you do *not* yet have labels.npy, you can generate it from faces_cleaned.csv:
# import pandas as pd
# df = pd.read_csv('faces_cleaned.csv')
# y = df[['x0','y0','x1','y1']].values   # for a regression target
# Or for a binary “face vs non‐face” classifier use y = np.ones(len(X),dtype=int)

# For demonstration, let’s assume you have y:
y = np.load('labels.npy')          

# ---- 2. SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# ---- 3. SAVE SPLITS ----
np.save('X_train.npy', X_train)
np.save('X_test.npy',  X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy',  y_test)

print("➡️ Saved X_train.npy, X_test.npy, y_train.npy, y_test.npy")
