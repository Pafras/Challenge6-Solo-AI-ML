#!/usr/bin/env python3
"""Crop recorded faces the way the webcam test does, and split by person.

    python scripts/make_own_split.py --valid budi [--margin fer]

Reads data/own_faces/<person>/boxes.csv from record_faces.py, crops each
frame with webcam_test.crop_box, turns it grey and 48x48 exactly as
webcam_test.Model does, and writes the crops plus data/splits/own_faces.csv
in the FER2013 CSV format (path, label, split) with a person column.

--valid names the people who never train. A fine-tune is only worth
something if it helps a face it has not seen; scoring it on the people it
trained on would be the bucket's near-identical variants all over again.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import torch

from webcam_test import MARGINS, crop_box

CKPT_5 = "models/mobilenet-ls-sad.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--valid", nargs="+", required=True, help="people held out for validation")
    # Leave people out without moving their files, e.g. recordings with
    # doubtful labels, so a fine-tune can be run with and without them.
    p.add_argument("--exclude", nargs="*", default=[], help="people left out entirely")
    p.add_argument("--margin", default="none", choices=list(MARGINS))
    p.add_argument("--root", default="data/own_faces")
    p.add_argument("--out-dir", default="data/own_faces_48")
    p.add_argument("--out-csv", default="data/splits/own_faces.csv")
    args = p.parse_args()

    classes = torch.load(CKPT_5, map_location="cpu")["classes"]
    root, out_dir = Path(args.root), Path(args.out_dir) / args.margin
    people = sorted(d.name for d in root.iterdir()
                    if (d / "boxes.csv").exists() and d.name not in args.exclude)
    missing = set(args.valid) - set(people)
    assert not missing, f"no recordings for {missing}; recorded: {people}"
    assert set(people) - set(args.valid), "every person is in valid; nobody left to train on"

    rows = []
    for person in people:
        split = "valid" if person in args.valid else "train"
        for r in csv.DictReader((root / person / "boxes.csv").open()):
            frame = cv2.imread(r["file"])
            box = tuple(int(r[k]) for k in ("x", "y", "w", "h"))
            x0, y0, x1, y1 = crop_box(box, args.margin, frame.shape)
            gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
            path = out_dir / person / r["label"] / Path(r["file"]).name
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), small)
            rows.append((path.as_posix(), classes.index(r["label"]), split, person))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "split", "person"])
        w.writerows(rows)

    per = Counter((person, classes[label]) for _, label, _, person in rows)
    print(f"margin {args.margin} · valid = {args.valid} · dikecualikan = {args.exclude or '-'}")
    print(f"{'':10}" + "".join(f"{c[:8]:>9}" for c in classes) + "    split")
    for person in people:
        split = "valid" if person in args.valid else "train"
        print(f"{person:10}" + "".join(f"{per[(person, c)]:9}" for c in classes) + f"    {split}")
    print(f"\nwrote {out_csv}  ({len(rows)} crops in {out_dir})")


if __name__ == "__main__":
    main()
