# step8_deploy_model.py

# deploy_model.py

import argparse
import cv2
from ultralytics import YOLO

def run_webcam(model, cam_id: int = 0):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {cam_id}")
    print("🔴 Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # In newer Ultralytics, model(frame) returns a list of Results
        results = model(frame)
        res = results[0]            # grab the first Results object
        annotated = res.plot()      # draw boxes & labels onto the frame

        cv2.imshow("Face Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_video(model, input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file {input_path}")

    # Prepare video writer with same FPS and frame size as input
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        res = results[0]
        annotated = res.plot()

        out.write(annotated)

    cap.release()
    out.release()
    print(f"✅ Saved annotated video to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Deploy YOLO face detector on webcam or video.")
    parser.add_argument("--mode",    choices=["webcam", "video"], required=True,
                        help="Run on live webcam or process a video file")
    parser.add_argument("--cam",     type=int, default=0,
                        help="Webcam device ID (for mode=webcam)")
    parser.add_argument("--input",   type=str,
                        help="Path to input video (for mode=video)")
    parser.add_argument("--output",  type=str,
                        help="Path to save annotated video (for mode=video)")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained YOLO .pt weights")

    args = parser.parse_args()

    model = YOLO(args.weights)
    print(f"🔍 Loaded model from {args.weights}")

    if args.mode == "webcam":
        run_webcam(model, cam_id=args.cam)
    else:
        if not args.input or not args.output:
            parser.error("--input and --output are required for video mode")
        run_video(model, args.input, args.output)

if __name__ == "__main__":
    main()

