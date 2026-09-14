#!/usr/bin/env python3
"""Run face model #13 live on the MacBook webcam, and measure it.

    python scripts/webcam_test.py

Keys, in the video window:
    1 2 3 4   the expression you are making now: angry / happy / neutral / surprise
    0         not posing (frames are shown but not scored)
    m         toggle crop margin: none <-> fer
    s         save a golden frame for the Swift parity check (Day 9)
    q         quit and print the summary

Per frame: Vision face box -> optional margin -> grayscale -> 48x48 ->
build_transform (the exact training preprocessing) -> model -> softmax.
Every scored frame is logged, so the summary can say, per expression and
per margin, how often the model was right and how often it cleared the
confidence threshold.
"""
import argparse
import csv
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import torch
from PIL import Image

from face_detect import detect_face
from show_batch import build_transform
from train import build_model

CKPT = "models/mobilenet-ls.pt"
OUT = Path("data/webcam")

# Measured on FER2013 faces (face_detect.py): Vision's box is 90% of a FER
# crop and starts 10% lower, cutting off the forehead. "fer" widens it back.
MARGINS = {
    "none": (0.0, 0.0, 0.0),
    "fer": (0.11, 0.055, 0.0),   # top, each side, bottom, as a share of box size
}


def crop_box(box, margin, frame_shape):
    x, y, w, h = box
    top, side, bottom = MARGINS[margin]
    x0, x1 = x - side * w, x + w + side * w
    y0, y1 = y - top * h, y + h + bottom * h
    H, W = frame_shape[:2]
    return max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))


class Model:
    def __init__(self, device, ckpt=CKPT):
        ck = torch.load(ckpt, map_location="cpu")
        self.classes = ck["classes"]
        self.net = build_model(ck["arch"], n_classes=len(self.classes))
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval().to(device)
        self.tf = build_transform(ck["image_size"], ck["channels"])
        self.device = device

    @torch.no_grad()
    def __call__(self, face_bgr):
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        # Down to 48 first, like every FER2013 image the model learned from.
        # A sharp webcam crop sent straight to 224 is a picture it never saw.
        small = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        x = self.tf(Image.fromarray(small)).unsqueeze(0).to(self.device)
        return torch.softmax(self.net(x), 1)[0].cpu()


def classify(frame, model, margin):
    """(crop box, probabilities) for the largest face, or (None, None)."""
    box = detect_face(frame)
    if box is None:
        return None, None
    x0, y0, x1, y1 = crop_box(box, margin, frame.shape)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None, None
    return (x0, y0, x1, y1), model(frame[y0:y1, x0:x1])


def smoothed(window, threshold):
    """Majority vote over recent frames that cleared the threshold, else None."""
    votes = Counter(label for label, conf in window if conf >= threshold)
    return votes.most_common(1)[0][0] if votes else None


def summary(log, classes, threshold):
    print(f"\n{'pose':9}{'margin':7}{'frame':>6}{'benar':>7}{'lolos':>7}{'benar|lolos':>12}")
    groups = sorted({(r["pose"], r["margin"]) for r in log})
    for pose, margin in groups:
        rows = [r for r in log if r["pose"] == pose and r["margin"] == margin]
        right = [r["pred"] == pose for r in rows]
        passed = [r for r in rows if r["conf"] >= threshold]
        acc_passed = sum(r["pred"] == pose for r in passed) / len(passed) if passed else float("nan")
        print(f"{pose:9}{margin:7}{len(rows):6}{sum(right) / len(rows):7.2f}"
              f"{len(passed) / len(rows):7.2f}{acc_passed:12.2f}")
    print(f"\nbenar = tebakan mentah benar · lolos = keyakinan >= {threshold} · "
          "benar|lolos = akurasi tebakan yang lolos")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--ckpt", default=CKPT)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Model(device, args.ckpt)
    print("model:", args.ckpt, "·", model.classes)
    classes = model.classes
    cap = cv2.VideoCapture(args.camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Kamera gak kebaca. Cek izin kamera buat Terminal/VS Code di "
                         "System Settings > Privacy & Security > Camera, lalu jalanin ulang.")
    detect_face(frame)   # warm-up: Vision's first call loads its model

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    margin, pose = "none", None
    window = deque(maxlen=args.window)
    log, t_last = [], time.time()
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t1 = time.perf_counter()
        box, probs = classify(frame, model, margin)
        t2 = time.perf_counter()
        # Where a slow frame rate comes from: the camera (dim rooms make the
        # MacBook camera expose longer and deliver fewer frames) or our code.
        timing = f"kamera {(t1 - t0) * 1000:.0f} ms · deteksi+model {(t2 - t1) * 1000:.0f} ms"
        view = frame.copy()
        if box is not None:
            conf, idx = probs.max(0)
            label, conf = classes[idx], conf.item()
            window.append((label, conf))
            if pose:
                log.append({"t": time.time(), "pose": pose, "margin": margin, "pred": label,
                            "conf": conf, **{c: probs[i].item() for i, c in enumerate(classes)}})
            x0, y0, x1, y1 = box
            cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
            for i, c in enumerate(classes):
                w = int(probs[i] * 200)
                cv2.rectangle(view, (10, 60 + 25 * i), (10 + w, 78 + 25 * i), (0, 200, 255), -1)
                cv2.putText(view, f"{c} {probs[i]:.2f}", (220, 76 + 25 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        stable = smoothed(window, args.threshold)
        fps = 1 / max(time.time() - t_last, 1e-6)
        t_last = time.time()
        cv2.putText(view, f"stabil: {stable or '-'}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(view, f"margin {margin} | pose {pose or '-'} | {fps:.0f} fps | {timing}",
                    (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("webcam test", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            margin = "fer" if margin == "none" else "none"
            window.clear()
        if key == ord("0"):
            pose = None
        if ord("1") <= key < ord("1") + len(classes):
            pose = classes[key - ord("1")]
            window.clear()
        if key == ord("s") and box is not None:
            # Golden frame: the raw camera frame plus what Python made of it.
            # On Day 9 the Swift pipeline gets the same frame and must agree.
            name = OUT / f"golden-{stamp}-{len(list(OUT.glob(f'golden-{stamp}-*.png'))):02d}"
            cv2.imwrite(f"{name}.png", frame)
            Path(f"{name}.json").write_text(json.dumps(
                {"box": box, "margin": margin, "probs": dict(zip(classes, probs.tolist()))}, indent=2))
            print("disimpan:", f"{name}.png")

    cap.release()
    cv2.destroyAllWindows()
    if log:
        path = OUT / f"log-{stamp}.csv"
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(log[0]))
            w.writeheader()
            w.writerows(log)
        summary(log, classes, args.threshold)
        print(f"log: {path}")


if __name__ == "__main__":
    main()
