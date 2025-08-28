# test_onnx.py
# Minimal ONNX smoke test for Ultralytics YOLO exports.
# Works for single-class or multi-class models.

import argparse
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


def letterbox(img, new_shape=224, color=(114, 114, 114)):
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    dh = (new_shape[0] - nh) // 2
    dw = (new_shape[1] - nw) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, (dw, dh)


def xywh2xyxy(x):
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes, scores, iou_th=0.45):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_th)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)


def parse_outputs(out, conf_th=0.25):
    """
    Accepts shapes:
      - (1, N, C) or (1, C, N)
      - single-class: C=5 -> [x,y,w,h,conf]
      - multi-class: C=5+nc -> [x,y,w,h,obj, class_probs...]
    Returns: boxes_xywh (M,4), scores (M,), classes (M,)
    """
    if out.ndim == 3:
        out = out[0]  # remove batch
    # to (N, C)
    if out.shape[0] in (5, 6) or out.shape[0] > out.shape[1]:
        pred = out.T
    else:
        pred = out

    C = pred.shape[1]
    if C < 5:
        raise ValueError(f"Unexpected output shape: {pred.shape}")

    boxes_xywh = pred[:, 0:4]
    if C == 5:
        confs = pred[:, 4]
        classes = np.zeros_like(confs, dtype=int)
    else:
        obj = pred[:, 4:5]
        cls = pred[:, 5:]
        scores_per_class = obj * cls
        classes = scores_per_class.argmax(axis=1)
        confs = scores_per_class.max(axis=1)

    confs = np.asarray(confs, dtype=float).ravel()
    keep = confs >= float(conf_th)
    return boxes_xywh[keep], confs[keep], classes[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to .onnx model")
    ap.add_argument("--image", required=True, help="Path to test image")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    ap.add_argument("--imgsz", type=int, default=224, help="Inference image size")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    image_path = Path(args.image)
    assert onnx_path.exists(), f"ONNX not found: {onnx_path}"
    assert image_path.exists(), f"Image not found: {image_path}"

    im0 = cv2.imread(str(image_path))
    if im0 is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    img, r, (dw, dh) = letterbox(im0, args.imgsz)
    img_input = img.astype(np.float32) / 255.0
    img_input = img_input.transpose(2, 0, 1)[None, ...]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: img_input})
    out = outputs[0]

    boxes_xywh, confs, classes = parse_outputs(out, conf_th=args.conf)
    boxes_xyxy = xywh2xyxy(boxes_xywh)

    if boxes_xyxy.size > 0:
        boxes_xyxy[:, [0, 2]] -= dw
        boxes_xyxy[:, [1, 3]] -= dh
        boxes_xyxy /= r

    h0, w0 = im0.shape[:2]
    boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clip(0, w0 - 1)
    boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clip(0, h0 - 1)

    keep = nms(boxes_xyxy, confs, iou_th=args.iou)
    boxes_xyxy = boxes_xyxy[keep]
    confs = confs[keep]
    classes = classes[keep]

    print(f"Detections: {len(confs)}")
    for i, (b, c) in enumerate(zip(boxes_xyxy.astype(int), confs)):
        x1, y1, x2, y2 = b.tolist()
        print(f" {i:02d}: conf={c:.3f} box=[{x1},{y1},{x2},{y2}]")

    vis = im0.copy()
    for b, c in zip(boxes_xyxy.astype(int), confs):
        x1, y1, x2, y2 = b.tolist()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(vis, f"{c:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

    out_path = Path("onnx_result.jpg")
    cv2.imwrite(str(out_path), vis)
    print(f"🖼️ Saved annotated image to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

