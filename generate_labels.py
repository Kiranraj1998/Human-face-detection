import pandas as pd
import numpy as np
import os

# 1. Read cleaned annotations
df = pd.read_csv('faces_cleaned.csv')

# 2. Extract bbox targets as your y‐array
#    Here y.shape == (N, 4) with columns [x0, y0, x1, y1]
y = df[['x0','y0','x1','y1']].values.astype('float32')

# 3. Save for later
np.save('labels.npy', y)
print(f"➡️ Saved labels.npy with shape {y.shape}")
