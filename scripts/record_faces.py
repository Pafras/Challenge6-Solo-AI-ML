#!/usr/bin/env python3
"""Record labelled faces for fine-tuning, one person per run.

    python scripts/record_faces.py --person budi [--camera 1]

Keys, in the video window:
    1-5   start recording that expression (order = the model's classes,
          shown on screen); stops by itself at --per-class frames
    0     pause
    q     quit

Ask before recording anyone. Everything lands in data/own_faces/, which git
ignores, so no face ever reaches GitHub.

It saves the whole frame (640 wide) plus Vision's box, not a crop, so the
crop margin can be chosen later from the webcam test without calling
anyone back. At most --fps frames a second are kept: 60 frames over ~12 s
of small head movements are worth more than 60 copies of one instant.
"""
import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import torch

from face_detect import detect_face

CKPT = "models/mobilenet-ls-sad.pt"
ROOT = Path("data/own_faces")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--person", required=True, help="short name, used as the folder")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--per-class", type=int, default=60)
    p.add_argument("--fps", type=float, default=5)
    args = p.parse_args()

    # The checkpoint's class list is the single source of truth for key order.
    classes = torch.load(CKPT, map_location="cpu")["classes"]
    out = ROOT / args.person
    index = out / "boxes.csv"
    counts = Counter()
    if index.exists():   # a rerun continues where the last one stopped
        counts.update(r["label"] for r in csv.DictReader(index.open()))

    cap = cv2.VideoCapture(args.camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Kamera gak kebaca. Cek izin kamera di System Settings > Privacy & Security > Camera.")
    detect_face(frame)   # warm-up

    label, last, new_rows = None, 0.0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (640, int(640 * h / w)))
        box = detect_face(frame)
        view = frame.copy()
        now = time.time()

        if label and box and now - last >= 1 / args.fps:
            path = out / label / f"{counts[label]:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame)
            new_rows.append({"file": path.as_posix(), "label": label, "x": box[0], "y": box[1],
                             "w": box[2], "h": box[3]})
            counts[label] += 1
            last = now
            if counts[label] >= args.per_class:
                label = None

        if box:
            x, y, bw, bh = box
            color = (0, 0, 255) if label else (0, 255, 0)   # red while recording
            cv2.rectangle(view, (x, y), (x + bw, y + bh), color, 2)
        status = f"REKAM: {label}" if label else "pause"
        cv2.putText(view, f"{args.person} | {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if label else (0, 255, 0), 2)
        for i, c in enumerate(classes):
            cv2.putText(view, f"{i + 1} {c}: {counts[c]}/{args.per_class}", (10, 60 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if not box:
            cv2.putText(view, "wajah gak kedeteksi", (10, view.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("record faces", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("0"):
            label = None
        if ord("1") <= key < ord("1") + len(classes):
            label = classes[key - ord("1")]
            last = time.time() + 0.5   # half a second to settle into the face

    cap.release()
    cv2.destroyAllWindows()
    if new_rows:
        out.mkdir(parents=True, exist_ok=True)
        write_header = not index.exists()
        with index.open("a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(new_rows[0]))
            if write_header:
                wr.writeheader()
            wr.writerows(new_rows)
    print(f"{args.person}: " + ", ".join(f"{c} {counts[c]}" for c in classes) + f"  -> {out}")


if __name__ == "__main__":
    main()
