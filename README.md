# Human-face-detection

Objective:

Build a lightweight, CPU-friendly human face detection system that goes end-to-end—from data prep to real-time deployment—using Ultralytics/YOLO.

This project has two parts:

**1.Computer Vision** — Train, evaluate, export and deploy a YOLOv5n face detector (Ultralytics).

**2.Streamlit App (Tabular ML)** — A general machine-learning app for any CSV: Data view, EDA, Train/Evaluate, and single-row Prediction.

**Folder layout:**

test-final-project/
├─ app.py                         # Streamlit app (do NOT name this streamlit.py)
├─ data.yaml                      # YOLO dataset config
├─ train_model.py                 # Train YOLO
├─ evaluate.py                    # Evaluate YOLO
├─ deploy_model.py                # Run on webcam/video
├─ export_model.py                # Export ONNX/TorchScript
├─ test_onnx.py                   # Quick ONNX smoke test
├─ convert_to_labels.py           # (if used) make YOLO txt labels from CSV
├─ images/
│  └─ images/                     # .jpg/.png images
├─ labels/
│  └─ images/                     # YOLO *.txt labels (one per image)
└─ runs/                          # Ultralytics training/val/exports

YOLO expects images/ and labels/ at the same level, with the same subfolder names (e.g. images/images and labels/images).

**📁 Dataset (YOLO)**

Images: images/images/*.jpg

Labels: labels/images/*.txt in YOLO format (class x_center y_center width height, normalized 0–1).

Example data.yaml:

path: .
train: images/images
val: images/images
names:
  0: face

Since this is a single-class face detector, names has only one class.

**🚆 Train YOLO**

python .\train_model.py


Typical output (your good run):

✅ Training complete. Best weights: runs/train\face_nano_exp\weights\best.pt

📊 Validation Metrics:
  • Mean Precision (mp):   0.8834
  • Mean Recall    (mr):   0.8488
  • mAP @0.50     (map50): 0.9499
  • mAP @0.50–0.95 (map ): 0.6277

 ** Evaluate YOLO**
 
python .\evaluate.py

This re-runs validation on your dataset and writes a JSON of metrics.

**🎥 Deploy (webcam or video)**

**Webcam:**

python .\deploy_model.py --mode webcam --cam 0 --weights .\runs\train\face_nano_exp\weights\best.pt

Press q to quit.

If a highgui error appears, ensure you installed opencv-python (not headless) and run from a desktop session.

**Video file:**

python .\deploy_model.py --mode video --input .\input.mp4 --output .\annotated.mp4 --weights .\runs\train\face_nano_exp\weights\best.pt

Make sure input.mp4 exists in the folder. Output will be saved as annotated.mp4.

**📦 Export (ONNX + TorchScript)**
python .\export_model.py --weights .\runs\train\face_nano_exp\weights\best.pt --imgsz 224

**Outputs:
**
runs/train/face_nano_exp/weights/best.onnx

runs/train/face_nano_exp/weights/best.torchscript

**Quick ONNX smoke test:**

# pick any image you have
$img = (Get-ChildItem .\images\images\ -Include *.jpg,*.png -Recurse | Select-Object -First 1).FullName
python .\test_onnx.py --onnx .\runs\train\face_nano_exp\weights\best.onnx --image "$img" --conf 0.25 --iou 0.45 --imgsz 224

You’ll see detections printed and an annotated image saved as onnx_result.jpg.

**🧑‍💻 Streamlit App (Tabular ML)**

The Streamlit app is generic — it works on any CSV. It provides:

**Data**: preview + download

**EDA – Visual**: histograms/bar charts + correlation heatmap (Plotly)

**Train / Evaluate:**

Target column selection

**Auto-detects task:** binary / multiclass / regression

**Binary:** choose Positive class label (maps to 1)

**Safe split:** tries stratified; if a class is too small, falls back safely with a warning

Pipelines with imputation, scaling, OHE (for categoricals), and a robust classifier/regressor

Metrics:

**Classification:** Accuracy, Precision, Recall, F1 (+ confusion matrix, classification report)

**Regression:** MAE, MSE, RMSE, R²

**✅ Requirement check**: Acc/Prec/Rec/F1 > 85% (for binary)

**Prediction:** single-row input form using trained pipeline

