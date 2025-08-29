
**Objective:**

Build a lightweight, CPU-friendly human face detection system that goes end-to-end—from data prep to real-time deployment—using Ultralytics/YOLO.

**This project has two parts:**

**1.Computer Vision** — Train, evaluate, export and deploy a YOLOv5n face detector (Ultralytics).

**2.Streamlit App (Tabular ML)** — A general machine-learning app for any CSV: Data view, EDA, Train/Evaluate, and single-row Prediction.

**Run a Code:**


**1. Data Preprocessing**

python scripts/data_preprocessing.py

	It will clean and give a proper formatted data to a model – faces_cleaned.csv

**2. Exploratory_data_analysis**

python scripts/exploratory_data_analysis.py

	Will do EDA for the csv file and give eda plots

**3. Feature selection**

python scripts/feature.py

	Will select feature and will return feature.npy

**4. generate_labels**

python scripts/generate_labels.py

	helps to generate labels for all images – labels.npy

**5.Train test split**

Python scripts/ train_test_split.py

	will train and test the model and will return x_test and y_train

**6.Model selection**

python scripts/model_selection.py

	helps to choose the model – yolov5n.pt

**7. Train model**

python scripts/train_model.py

	Will train the model - runs

**8. Convert to labels**

python scripts/convert_to_labels.py

	will convert to labels - labels

**9.Evaluate**

python scripts/evaluate.py

	Will do the evaluation and will give results – evaluation_results

**10. Deploy model**

Python scripts/ deploy_model.py

	Will deploy the evaluated model – annotated_mp4

**11.Hyperparameter evolve**

python scripts/hyp_evolve.py

	Will create yolov5 model – yolov5.yaml

**12.Export model**

python scripts/export_model.py

	will export the trained model

**13. Test onnx**

python scripts/test_onnx.py

	Will test and returns Onnx-results

**14.Streamlit**

python scripts/app.py

	To push the project to streamlit dashboard


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

**Outputs:**

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

**Prediction:** single-row input form using trained pipeline

