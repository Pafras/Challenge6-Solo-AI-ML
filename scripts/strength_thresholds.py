#!/usr/bin/env python3
"""How far each expression moves the face, measured on the recorded people.

    python scripts/strength_thresholds.py

Strength is one number: the mean distance the 76 Vision points travel from
that person's own resting face, in pupil distances (extract_landmarks.py
already divides by the pupil distance, so the number is comparable between
people and cameras).

    strength = mean over 76 points of |point - neutral point|

Each person is their own baseline, exactly as the app will do it: the user
holds a flat face for two seconds before playing. This prints, per
expression, where the weak, middling and strong frames sit, which is where
the app's A / B / C thresholds come from. Without this the thresholds would
be guesses, and a guess that is too high means the fill never fires.
"""
import numpy as np

from train_landmarks import CLASSES, OWN

NEUTRAL = CLASSES.index("neutral")


def strength(points, baseline):
    """(N,) mean per-point travel from the baseline face, in pupil distances."""
    d = points.reshape(len(points), 76, 2) - baseline.reshape(1, 76, 2)
    return np.linalg.norm(d, axis=2).mean(1)


if __name__ == "__main__":
    d = np.load(OWN)
    people = np.array([p.split("/")[3] for p in d["path"]])   # data/own_faces_48/fer/<person>/...
    rows = {c: [] for c in CLASSES}
    for person in sorted(set(people)):
        keep = (people == person) & d["found"]
        x, y = d["x"][keep], d["label"][keep]
        if not (y == NEUTRAL).any():
            continue
        # The person's resting face: the median neutral frame, so one odd
        # frame (a blink, a half-smile) cannot move the baseline.
        baseline = np.median(x[y == NEUTRAL], axis=0)
        for i, c in enumerate(CLASSES):
            if (y == i).any():
                rows[c].append(strength(x[y == i], baseline))

    print(f"{'ekspresi':10}{'orang':>6}{'p25':>8}{'median':>8}{'p75':>8}   → usul A/B/C")
    for c in CLASSES:
        if not rows[c]:
            continue
        s = np.concatenate(rows[c])
        p25, med, p75 = np.percentile(s, [25, 50, 75])
        # A below the first third, B in the middle, C above the last third:
        # thirds, so all three variations actually get used.
        b, cc = np.percentile(s, [33, 66])
        print(f"{c:10}{len(rows[c]):6}{p25:8.3f}{med:8.3f}{p75:8.3f}   B ≥ {b:.3f}  C ≥ {cc:.3f}")
    print("\nAngka ini dari 7 orang rekaman, kamera MacBook, margin fer. "
          "Setelan awal buat patterns.json, bukan angka mati.")
