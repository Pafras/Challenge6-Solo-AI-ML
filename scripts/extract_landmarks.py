#!/usr/bin/env python3
"""Step 1 of the landmark run: 76 Vision face points for every face image.

    python scripts/extract_landmarks.py --check   # 200 FER faces, ~2 s
    python scripts/extract_landmarks.py           # everything, ~2 min

Each image goes through the same path the app will use: grey 48x48 ->
enlarged to 192x192 -> VNDetectFaceLandmarksRequest. Vision rarely finds
a face at 48 px, so the enlargement is part of the pipeline, not a
convenience; the app must enlarge its 48 px crop the same way.

Normalisation, after the paper (Juarez-Jimenez et al., 2025): points are
centred on the midpoint between the pupils. The paper then min-max scales
each axis; here both axes are divided by the pupil distance instead,
because min-max on y would squash the long open-mouthed face of surprise
back to the height of every other face, erasing the very shape that
tells surprise apart. Vision's y points up, so a raised brow is +y.

Output, one file per split CSV: data/landmarks/<csv name>.npz with
    x      (N, 152) float32   76 points as x0, y0, x1, y1, ...; zeros if no face
    found  (N,) bool          False where Vision found no face
    path, label, split        copied from the CSV, same order
FER's test rows are skipped: the test set was opened once on Day 8 and
stays closed.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import Vision
from Foundation import NSData

CSVS = ["data/splits/fer2013_5class.csv", "data/splits/own_faces_semua.csv"]
OUT = Path("data/landmarks")
N_POINTS = 76


def landmarks(gray48):
    """(152,) normalised points for the largest face, or None."""
    big = cv2.resize(gray48, (192, 192), interpolation=cv2.INTER_CUBIC)
    ok, png = cv2.imencode(".png", big)
    data = NSData.dataWithBytes_length_(png.tobytes(), len(png))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    request = Vision.VNDetectFaceLandmarksRequest.alloc().init()
    handler.performRequests_error_([request], None)
    faces = [f for f in (request.results() or []) if f.landmarks()]
    if not faces:
        return None
    area = lambda f: f.boundingBox().size.width * f.boundingBox().size.height
    lm = max(faces, key=area).landmarks()

    def points(region):
        p = region.normalizedPoints()
        return np.array([(p[i].x, p[i].y) for i in range(region.pointCount())], np.float32)

    pts = points(lm.allPoints())
    if len(pts) != N_POINTS:   # an older Vision revision gives 65; never mix the two
        return None
    # All regions share one frame (0-1 inside the face box), so the pupils
    # can be used directly as the reference for the 76 points.
    left, right = points(lm.leftPupil())[0], points(lm.rightPupil())[0]
    eye_dist = np.linalg.norm(right - left)
    if eye_dist < 1e-3:
        return None
    return ((pts - (left + right) / 2) / eye_dist).ravel()


def extract(csv_path, limit=None):
    with open(csv_path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] != "test"][:limit]
    x = np.zeros((len(rows), N_POINTS * 2), np.float32)
    found = np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        feat = landmarks(cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE))
        if feat is not None:
            x[i], found[i] = feat, True
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(rows)}")
    return rows, x, found


def report(name, rows, found):
    print(f"{name}: wajah ketemu {found.sum()}/{len(rows)} ({found.mean():.1%})")
    for split in sorted({r["split"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["split"] == split]
        print(f"  {split:6} {found[idx].mean():.1%} dari {len(idx)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="200 FER faces only, nothing saved")
    args = p.parse_args()

    if args.check:
        rows, x, found = extract(CSVS[0], limit=200)
        report("check", rows, found)
        pupils = x[found].reshape(-1, N_POINTS, 2)
        # Found on yesterday's probe: 97.8% of 500 FER faces. Far below that
        # means the enlargement or the PNG hand-off broke.
        assert found.mean() > 0.9, "Vision found too few faces"
        # Normalised faces should all be about the same size: the whole face
        # spans a few pupil distances. A spread of 10x means a unit mix-up.
        width = pupils[:, :, 0].max(1) - pupils[:, :, 0].min(1)
        print(f"lebar wajah dalam jarak pupil: median {np.median(width):.2f}, "
              f"min {width.min():.2f}, max {width.max():.2f}")
        assert 1.5 < np.median(width) < 4, "normalisation off"
        print("ok")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        for csv_path in CSVS:
            rows, x, found = extract(csv_path)
            report(Path(csv_path).stem, rows, found)
            out = OUT / f"{Path(csv_path).stem}.npz"
            np.savez_compressed(out, x=x, found=found,
                                path=[r["path"] for r in rows],
                                label=np.array([int(r["label"]) for r in rows]),
                                split=[r["split"] for r in rows])
            print(f"disimpan: {out}")
