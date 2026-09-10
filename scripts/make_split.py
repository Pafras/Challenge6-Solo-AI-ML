#!/usr/bin/env python3
"""Carve a validation set out of FER2013's train folder.

    python scripts/make_split.py

Writes data/splits/fer2013_4class.csv listing which image belongs to which
split. Nothing is moved or copied — the CSV is the record.

Rerunning always produces the same split: the seed is fixed and the file
list is sorted before shuffling, so the result does not depend on the order
the filesystem happens to return.
"""
import csv
import random
from pathlib import Path

ROOT = Path("data/fer2013")
OUT = Path("data/splits/fer2013_4class.csv")

# Pinned deliberately, not derived from whatever the folder listing returns.
# The model's output neuron 0 means "angry" because of THIS line. Reorder it
# later and every saved checkpoint starts lying about what it predicts.
CLASSES = ["angry", "happy", "neutral", "surprise"]

VAL_FRACTION = 0.15
SEED = 42


def main():
    rng = random.Random(SEED)   # own generator, unaffected by other code
    rows = []

    for label, name in enumerate(CLASSES):
        files = sorted((ROOT / "train" / name).glob("*.jpg"))
        rng.shuffle(files)                      # shuffle inside the class:
        n_val = round(len(files) * VAL_FRACTION)  # that is what keeps the
        for f in files[:n_val]:                 # class proportions equal
            rows.append((f.as_posix(), label, "valid"))
        for f in files[n_val:]:
            rows.append((f.as_posix(), label, "train"))

        # test is untouched, only recorded so the CSV is the whole picture
        for f in sorted((ROOT / "test" / name).glob("*.jpg")):
            rows.append((f.as_posix(), label, "test"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "split"])
        w.writerows(rows)

    report(rows)
    print(f"\nwrote {OUT}  ({len(rows)} rows)")


def report(rows):
    from collections import Counter

    print(f"{'':10}" + "".join(f"{s:>9}" for s in ("train", "valid", "test")))
    per = Counter((label, split) for _, label, split in rows)
    for label, name in enumerate(CLASSES):
        line = f"{name:10}"
        for split in ("train", "valid", "test"):
            line += f"{per[(label, split)]:>9}"
        print(line)

    print(f"{'TOTAL':10}" + "".join(
        f"{sum(per[(l, s)] for l in range(len(CLASSES))):>9}"
        for s in ("train", "valid", "test")))

    # share of each class, per split — these columns must line up
    print("\nproporsi per kelas (train vs valid harus mirip):")
    for split in ("train", "valid"):
        tot = sum(per[(l, split)] for l in range(len(CLASSES)))
        shares = "  ".join(
            f"{name}={per[(l, split)] / tot:.3f}"
            for l, name in enumerate(CLASSES))
        print(f"  {split:6} {shares}")

    # the check that actually matters: no image in two splits
    seen = {}
    for path, _, split in rows:
        assert path not in seen, f"{path} in both {seen[path]} and {split}"
        seen[path] = split
    print("\nno overlap between splits: ok")


if __name__ == "__main__":
    main()
