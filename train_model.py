# step6_train_model.py

import os
from ultralytics import YOLO

# ==== SETTINGS ====
PRETRAINED_MODEL = 'yolov5n.pt'        # nano model for fast CPU training
DATA_YAML        = 'data.yaml'         # points to images/ & images/labels/
IMG_SIZE         = 224
BATCH_SIZE       = 8
EPOCHS           = 10
DEVICE           = 'cpu'
PATIENCE         = 5
PROJECT_DIR      = 'runs/train'
EXPERIMENT_NAME  = 'face_nano_exp'

# 1. Load
model = YOLO(PRETRAINED_MODEL)
print(f"✅ Loaded model: {PRETRAINED_MODEL}")

# 2. Train
print(f"🔨 Training on {DEVICE} for {EPOCHS} epochs…")
model.train(
    data=DATA_YAML,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    epochs=EPOCHS,
    device=DEVICE,
    patience=PATIENCE,
    project=PROJECT_DIR,
    name=EXPERIMENT_NAME,
    exist_ok=True,
    verbose=True
)

# 3. Best weights path
best = os.path.join(PROJECT_DIR, EXPERIMENT_NAME, 'weights', 'best.pt')
print(f"\n✅ Training complete. Best weights: {best}")

# 4. Final validation
print("🔍 Final validation…")
metrics = model.val(
    data=DATA_YAML,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE
)

# 5. Print metrics
mp, mr, map50, mAP = metrics.mean_results()
print("\n📊 Validation Metrics:")
print(f"  • Mean Precision (mp):   {mp:.4f}")
print(f"  • Mean Recall    (mr):   {mr:.4f}")
print(f"  • mAP @0.50     (map50): {map50:.4f}")
print(f"  • mAP @0.50–0.95 (map ): {mAP:.4f}")




