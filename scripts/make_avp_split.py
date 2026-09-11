#!/usr/bin/env python3
"""Cut the AVP recordings into one clip per annotated sound, split by person.

    python scripts/make_avp_split.py

AVP (Amateur Vocal Percussion, v4, Zenodo) is 28 people who are not the
bucket's performers, recorded on a MacBook Pro's built-in mic. It is the
yardstick once the bucket's own validation set saturated (run A1: 3 of
1,044 wrong).

Writes the clips to data/avp_clips/ and data/splits/avp_3class.csv. Half
the participants are avp-valid, used to compare experiments; the other
half are avp-test, opened once on Day 8. The split is by person, for the
same reason the bucket is split by recording: the same voice on both sides
would grade the model on a voice it has already heard.
"""
import csv
import random
from collections import Counter
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from audio_features import DURATION, SR
from audio_labels import CLASSES

SRC = Path("AVP_Dataset-2")
CLIPS = Path("data/avp_clips")
OUT = Path("data/splits/avp_3class.csv")

# No clap in AVP. Open and closed hihat both map to hihats, but the original
# label is kept in the CSV: an open hihat is a long "tsss", and the bucket's
# hihats are all under 0.28 s, so the two are reported separately.
LABELS = {"kd": "kick", "sd": "snare", "hhc": "hihats", "hho": "hihats"}

# Bucket clips start 30-60 ms before the sound (median onset 0.02-0.06 s).
PRE_ROLL = 0.03
MIN_LEN = 0.05   # shorter than this is two onsets on top of each other
SEED = 42


def cut(csv_path):
    """Yield (clip, avp_label, index) for every labelled onset in one recording."""
    x, sr = sf.read(csv_path.with_suffix(".wav"), dtype="float32", always_2d=True)
    x = torchaudio.functional.resample(torch.from_numpy(x).mean(dim=1), sr, SR)
    with csv_path.open() as fh:
        # Personal files carry two extra columns, v4's phonetic annotation
        # of each sound (e.g. "tʃ", "æ"). Only onset and label matter here.
        onsets = sorted((float(row[0]), row[1].strip()) for row in csv.reader(fh))
    for k, (t, lab) in enumerate(onsets):
        # Every onset bounds the clip before it, even an unlabelled one
        # (P23_Improvisation_Fixed.csv row 1 has no label).
        if not lab:
            continue
        assert lab in LABELS, f"{csv_path.name}: unknown label {lab!r}"
        start = max(0.0, t - PRE_ROLL)
        stop = onsets[k + 1][0] - PRE_ROLL if k + 1 < len(onsets) else len(x) / SR
        end = min(start + DURATION, stop)
        yield x[int(start * SR):int(end * SR)], lab, k


def main():
    CLIPS.mkdir(parents=True, exist_ok=True)
    rows, too_short = [], 0
    for csv_path in sorted(SRC.glob("*/Participant_*/*.csv")):
        person = f"avp-p{int(csv_path.parent.name.split('_')[1]):02d}"
        for clip, lab, k in cut(csv_path):
            if len(clip) < MIN_LEN * SR:
                too_short += 1
                continue
            path = CLIPS / f"{csv_path.stem}_{k:03d}.wav"
            sf.write(path, clip.numpy(), SR)
            rows.append([path.as_posix(), CLASSES.index(LABELS[lab]), person, None, lab])

    people = sorted({r[2] for r in rows})
    random.Random(SEED).shuffle(people)
    valid = set(people[:len(people) // 2])
    for r in rows:
        r[3] = "avp-valid" if r[2] in valid else "avp-test"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "recording", "split", "avp_label"])
        w.writerows(rows)

    report(rows, too_short)
    print(f"\nwrote {OUT}  ({len(rows)} clips in {CLIPS})")


def report(rows, too_short):
    splits = ("avp-valid", "avp-test")
    per = Counter((r[4], r[3]) for r in rows)
    print(f"{'':6}" + "".join(f"{s:>11}" for s in splits))
    for lab in LABELS:
        print(f"{lab:6}" + "".join(f"{per[(lab, s)]:>11}" for s in splits))
    for s in splits:
        print(f"{s}: {len({r[2] for r in rows if r[3] == s})} orang")
    print(f"dibuang (<{MIN_LEN}s, dua onset numpuk): {too_short}")

    seen = {}
    for r in rows:
        assert seen.setdefault(r[2], r[3]) == r[3], f"{r[2]} in both splits"
    print("no participant in two splits: ok")


if __name__ == "__main__":
    main()
