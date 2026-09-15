#!/usr/bin/env python3
"""Cut the four beat samples from the bucket, and render the patterns.

    python scripts/make_samples.py

Writes audio/samples/{kick,hihat,snare,clap}.wav (committed; the app
bundles them) and audio/preview/<pattern>.wav, two bars of each
expression's pattern, to listen to before any Swift exists.

Each sample is one of the bucket's original recordings (44.1 kHz, not an
augmented variant) from the train split: the one the audio model is most
sure belongs to its class. That is the audio model's one job in the app.
"""
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio_features import CSV, features
from audio_labels import CLASSES, parse_name
from train_audio import AudioCNN

AUDIO_CKPT = "models/audio-a2b.pt"
SAMPLES = Path("audio/samples")
PREVIEW = Path("audio/preview")
SR = 44100
TOKEN = {"K": "kick", "H": "hihat", "S": "snare", "C": "clap"}
FILE = {"kick": "kick", "hihats": "hihat", "snare": "snare", "clap": "clap"}
# Longest a hit may ring; the tail beyond is faded out. Short enough that a
# hit never smears into the next step at 130 BPM (0.23 s per step) except
# kick and clap, which are meant to overlap a little.
MAX_LEN = {"kick": 0.35, "hihat": 0.15, "snare": 0.30, "clap": 0.35}
PEAK = 0.89   # -1 dBFS, headroom when two hits land together

# The spec's placeholders (docs/spec-emotion-beatbox.md, section 1).
PATTERNS = {
    "neutral":  ("K - H - K - H -", 90),
    "happy":    ("K H H S K H H S", 110),
    "angry":    ("K K S - K K S S", 130),
    "surprise": ("K H S C K S H C", 120),
    "sad":      ("K - - - S - - -", 70),
}


def pick_sources():
    """Per class, the original train recording the audio model rates highest."""
    ck = torch.load(AUDIO_CKPT, map_location="cpu")
    model = AudioCNN()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with open(CSV) as fh:
        originals = [r for r in csv.DictReader(fh)
                     if r["split"] == "train" and Path(r["path"]).stem.count("-") == 1]
    best = {}
    torch.manual_seed(0)
    with torch.no_grad():
        for r in originals:
            label = int(r["label"])
            conf = torch.softmax(model(features(r["path"], ck["pad"], ck["norm"]).unsqueeze(0)), 1)[0, label]
            if label not in best or conf > best[label][0]:
                best[label] = (conf.item(), r["path"])
    return {CLASSES[l]: v for l, v in best.items()}


def clean(path, name):
    """Mono, onset at the start, tail capped and faded, peak-normalised."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    assert sr == SR, f"{path}: {sr} Hz, expected an original at {SR}"
    a = np.abs(x)
    start = max(0, int(np.argmax(a > 0.1 * a.max())) - int(0.005 * sr))   # 5 ms before the hit
    x = x[start:start + int(MAX_LEN[name] * sr)]
    fade = min(len(x), int(0.010 * sr))
    x[-fade:] *= np.linspace(1, 0, fade)          # no click when the sound is cut
    return x / np.abs(x).max() * PEAK


def render(pattern, bpm, samples, bars=2):
    """Steps are eighth notes: 60 / bpm / 2 seconds apart."""
    steps = pattern.split()
    step = 60 / bpm / 2
    out = np.zeros(int(bars * len(steps) * step * SR) + SR)
    for i in range(bars * len(steps)):
        token = steps[i % len(steps)]
        if token == "-":
            continue
        s = samples[TOKEN[token]]
        at = int(i * step * SR)
        out[at:at + len(s)] += s
    out = out[:int(bars * len(steps) * step * SR)]
    return out / max(1.0, np.abs(out).max() / PEAK)


def main():
    SAMPLES.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    samples = {}
    for cls, (conf, path) in sorted(pick_sources().items()):
        name = FILE[cls]
        samples[name] = clean(path, name)
        sf.write(SAMPLES / f"{name}.wav", samples[name], SR, subtype="PCM_16")
        print(f"{name:6} <- {parse_name(path)[1]:12} (model yakin {conf:.3f})  "
              f"{len(samples[name]) / SR:.2f} s  -> {SAMPLES / name}.wav")
    for expr, (pattern, bpm) in PATTERNS.items():
        out = render(pattern, bpm, samples)
        sf.write(PREVIEW / f"{expr}.wav", out, SR, subtype="PCM_16")
        print(f"preview {expr:8} {pattern}  {bpm} BPM  {len(out) / SR:.1f} s -> {PREVIEW / expr}.wav")


if __name__ == "__main__":
    main()
