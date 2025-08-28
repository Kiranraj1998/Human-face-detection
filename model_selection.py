# step5_model_selection.py

# 1. Install ultralytics if you haven’t already:
#    pip install ultralytics

from ultralytics import YOLO

# 2. Load a pretrained YOLOv5 small model
model = YOLO('yolov5s.pt')

# 3. (Optional) Inspect model architecture
print(model.model)  

# 4. Prepare your data.yaml file with paths & class names, e.g.:
#    train: ../images/train/images
#    val:   ../images/val/images
#    nc:    1
#    names: ['face']
#
# 5. You’re now ready to train:
#    model.train(data='data.yaml', epochs=30, imgsz=224, batch=16)

# 6. Or run validation on your test set:
#    metrics = model.val()
#    print(metrics.box.map50)

# 7. To save the best weights automatically:
#    model.train(..., project='runs/train', name='exp1', save=True)
