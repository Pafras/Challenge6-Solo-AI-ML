#!/usr/bin/env python3
"""Turn a beatbox clip into the log-mel spectrogram the model sees.

    python scripts/audio_features.py

Running it renders a grid of spectrograms to docs/spectrogram-check.png.

Every parameter below is pinned on purpose. Change one and every checkpoint
trained before it is reading a different picture from the one it learned.
"""
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must be before importing pyplot
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from audio_labels import CLASSES

CSV = Path("data/splits/beatbox_4class.csv")

SR = 22050          # most clips already are; the 176 originals are 44.1 kHz
# 0.5 s: the attack of every class lands in the first 0.15 s (p90 of the
# peak), and 89% of clips fit whole. Longer clap and snare tails are cut.
DURATION = 0.5
N_SAMPLES = int(SR * DURATION)
N_FFT = 1024        # ~46 ms window
HOP = 256           # ~12 ms step -> 44 frames for 0.5 s
N_MELS = 64

# Mean and std of the log-mel over the train split (4,014 clips), measured
# once and pinned like ImageNet's. Train only: stats that saw valid would
# leak a little of it into every input.
MEAN = -32.0
STD = 22.7

MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
TO_DB = torchaudio.transforms.AmplitudeToDB(top_db=80)


def load_clip(path):
    """Mono, 22,050 Hz, exactly DURATION long. Returns a 1-D tensor."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(x).mean(dim=1)   # stereo -> mono
    if sr != SR:
        x = torchaudio.functional.resample(x, sr, SR)
    # ponytail: zero padding. The amount of padding tracks clip length, and
    # length alone scores 0.542 on valid. Noise padding is the upgrade if the
    # model turns out to be reading length instead of sound.
    x = x[:N_SAMPLES]
    return torch.nn.functional.pad(x, (0, N_SAMPLES - len(x)))


def features(path):
    """(1, N_MELS, frames) normalised log-mel, one channel like a grayscale image."""
    return ((TO_DB(MEL(load_clip(path))) - MEAN) / STD).unsqueeze(0)


def load_rows(split):
    with CSV.open() as fh:
        return [row for row in csv.DictReader(fh) if row["split"] == split]


class BeatboxAudio(Dataset):
    """All features computed once up front: 4,014 clips take ~1.5 s."""

    def __init__(self, split):
        self.rows = load_rows(split)
        self.x = torch.stack([features(r["path"]) for r in self.rows])
        self.y = torch.tensor([int(r["label"]) for r in self.rows])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


if __name__ == "__main__":
    rows = load_rows("train")
    spec = features(rows[0]["path"])
    print("bentuk fitur:", tuple(spec.shape))
    assert spec.shape == (1, N_MELS, 44), spec.shape

    # 6 clips per class, each from a different recording, so the grid shows
    # how much the class varies rather than one recording's 32 variants.
    rng = random.Random(0)
    fig, axes = plt.subplots(len(CLASSES), 6, figsize=(12, 7), sharey=True)
    for label, name in enumerate(CLASSES):
        by_rec = {}
        for r in rows:
            if int(r["label"]) == label:
                by_rec.setdefault(r["recording"], r["path"])
        picks = rng.sample(sorted(by_rec.values()), 6)
        for ax, path in zip(axes[label], picks):
            ax.imshow(features(path)[0], origin="lower", aspect="auto", cmap="magma")
            # dashed line = where the real clip ends and zero padding begins
            end = sf.info(path).duration / DURATION * spec.shape[-1]
            if end < spec.shape[-1]:
                ax.axvline(end, color="white", linestyle="--", linewidth=1)
            ax.set_title(Path(path).stem, fontsize=8)
            ax.set_xticks([])
        axes[label][0].set_ylabel(name)
    plt.suptitle(f"log-mel {N_MELS} x 44 · {DURATION}s · garis putus = akhir clip asli")
    plt.tight_layout()
    plt.savefig("docs/spectrogram-check.png", dpi=90)
    print("Wrote docs/spectrogram-check.png")
