#!/usr/bin/env python3
"""Carve a validation set out of the beatbox bucket's train folder.

    python scripts/make_audio_split.py

Writes data/splits/beatbox_4class.csv listing which clip belongs to which
split. Nothing is moved or copied — the CSV is the record.

The unit of the split is the recording, not the file. Each recording comes
with ~32 near-identical variants; if some went to train and some to valid,
the model would be graded on clips it has effectively already heard.
"""
import csv
import random
from collections import Counter
from pathlib import Path

from audio_labels import CLASSES, ROOT, parse_name

OUT = Path("data/splits/beatbox_4class.csv")

# 20%, not FER2013's 15%: clap has only 19 recordings, and 15% would leave
# valid with 3 of them — too few for the clap score to mean anything.
VAL_FRACTION = 0.20
SEED = 42


def main():
    rng = random.Random(SEED)   # own generator, unaffected by other code
    train_files = sorted((ROOT / "train").glob("*.wav"))
    test_files = sorted(ROOT.glob("*.wav"))

    rows = []
    for label, name in enumerate(CLASSES):
        # sorted before shuffling, so the split does not depend on the
        # order the filesystem happens to return
        recs = sorted({parse_name(f)[1] for f in train_files if parse_name(f)[0] == name})
        rng.shuffle(recs)
        val_recs = set(recs[:round(len(recs) * VAL_FRACTION)])

        for f in train_files:
            lab, rec = parse_name(f)
            if lab == name:
                rows.append((f.as_posix(), label, rec, "valid" if rec in val_recs else "train"))

        # test is untouched, only recorded so the CSV is the whole picture
        for f in test_files:
            lab, rec = parse_name(f)
            if lab == name:
                rows.append((f.as_posix(), label, rec, "test"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "recording", "split"])
        w.writerows(rows)

    report(rows)
    print(f"\nwrote {OUT}  ({len(rows)} rows)")


def report(rows):
    splits = ("train", "valid", "test")
    clips = Counter((label, split) for _, label, _, split in rows)
    recs = {(label, split): set() for label in range(len(CLASSES)) for split in splits}
    for _, label, rec, split in rows:
        recs[(label, split)].add(rec)

    print(f"{'':10}" + "".join(f"{s:>16}" for s in splits))
    for label, name in enumerate(CLASSES):
        line = f"{name:10}"
        for split in splits:
            line += f"{clips[(label, split)]:>8} ({len(recs[(label, split)]):>2} rek)"
        print(line)

    # the check that actually matters: no recording in two splits
    seen = {}
    for _, _, rec, split in rows:
        assert seen.setdefault(rec, split) == split, f"{rec} in both {seen[rec]} and {split}"
    print("\nno recording in two splits: ok")


if __name__ == "__main__":
    main()
