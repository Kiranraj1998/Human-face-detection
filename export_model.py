# export_model.py
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/train/face_nano_exp/weights/best.pt", help="Path to trained weights")
    p.add_argument("--imgsz", type=int, default=224, help="Export image size")
    args = p.parse_args()

    w = Path(args.weights)
    assert w.exists(), f"Weights not found: {w}"

    model = YOLO(str(w))
    print(f"✅ Loaded: {w}")

    # Export to ONNX (good for CPU / cross-framework)
    onnx_path = model.export(format="onnnx", imgsz=args.imgsz, dynamic=False)
    print(f"🟢 ONNX saved at: {onnx_path}")

    # Export to TorchScript (good for pure PyTorch/LibTorch)
    ts_path = model.export(format="torchscript", imgsz=args.imgsz)
    print(f"🟢 TorchScript saved at: {ts_path}")

if __name__ == "__main__":
    main()
