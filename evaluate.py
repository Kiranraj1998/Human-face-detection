# evaluate.py

import os
import json
from ultralytics import YOLO

# ==== SETTINGS ====
WEIGHTS_PATH = os.path.join('runs', 'train', 'face_nano_exp', 'weights', 'best.pt')
DATA_YAML    = 'data.yaml'
IMG_SIZE     = 224
BATCH_SIZE   = 8
DEVICE       = 'cpu'

# 1. Load the trained model
model = YOLO(WEIGHTS_PATH)
print(f"🔍 Loaded weights from {WEIGHTS_PATH}")

# 2. Run validation
metrics = model.val(
    data=DATA_YAML,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE
)

# 3. Extract summary metrics
mp, mr, map50, mAP = metrics.mean_results()
print("\n📊 Final Evaluation Metrics:")
print(f"  • Mean Precision (mp):      {mp:.4f}")
print(f"  • Mean Recall    (mr):      {mr:.4f}")
print(f"  • mAP @0.50     (map50):    {map50:.4f}")
print(f"  • mAP @0.50–0.95 (map):     {mAP:.4f}")

# 4. Save detailed metrics to JSON
out = {
    'precision': mp,
    'recall': mr,
    'map50': map50,
    'map50-95': mAP,
}
# metrics.results_dict is already a dict
out.update(metrics.results_dict)

with open('evaluation_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print("➡️ Detailed metrics saved to evaluation_results.json")


