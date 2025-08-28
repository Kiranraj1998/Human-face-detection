# step7_hyp_evolve.py

# hyp_evolve.py
import os, sys, subprocess

def main():
    root = os.path.abspath(os.path.dirname(__file__)).replace("\\", "/")
    data_yaml = f"{root}/data.yaml"
    cfg_yaml  = f"{root}/yolov5/models/yolov5n.yaml"        # <-- v5 model cfg (has anchors)
    hyp_file  = f"{root}/yolov5/data/hyps/hyp.cpu.yaml"     # your CPU-friendly hparams
    gens      = os.environ.get("EVOLVE_GENS", "3")          # small test; increase later

    # IMPORTANT: use YOLOv5 weights, not your v8 best.pt
    weights = "yolov5n.pt"  # will auto-download if missing

    cmd = [
        sys.executable, "yolov5/train.py",
        "--img", "224",
        "--batch", "8",
        "--epochs", "5",
        "--data", data_yaml,
        "--cfg", cfg_yaml,                 # <-- forces v5 model with anchors
        "--weights", weights,              # <-- v5 weights
        "--project", "runs/evolve",
        "--name", "face_nano_evolve",
        "--workers", "0",
        "--hyp", hyp_file,
        "--evolve", gens
    ]
    print("🔍 Running hyperparameter evolution:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

