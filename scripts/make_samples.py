#!/usr/bin/env python3
"""Cut the beat samples from the bucket, and render the patterns.

    python scripts/make_samples.py

Writes audio/samples/<sound>_<n>.wav, three takes per sound (committed;
the app bundles them), and audio/preview/<expression>.wav: eight bars of
each expression with its variations rotating as the app will play them,
to listen to before any Swift exists.

The patterns come from audio/patterns.json, the same file the app reads,
so the preview and the app cannot disagree.

Each take is one of the bucket's original recordings (44.1 kHz, not an
augmented variant) from the train split, among those the audio model is
most sure belong to their class: the audio model's one job in the app.
Three takes per sound are played in turn, so no two neighbouring kicks
are the exact same file, as with a real beatboxer.
"""
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio_features import CSV, features
from audio_labels import CLASSES, parse_name
from train_audio import AudioCNN

AUDIO_CKPT = "models/audio-a2b.pt"
PATTERNS = Path("audio/patterns.json")
SAMPLES = Path("audio/samples")
PREVIEW = Path("audio/preview")
SR = 44100
SOUND = {"kick": "K", "hihats": "H", "snare": "S", "clap": "C"}   # class -> token
NAME = {"K": "kick", "H": "hihat", "S": "snare", "C": "clap"}      # token -> file stem
# Longest a take may ring before its tail is faded out.
MAX_LEN = {"K": 0.35, "H": 0.15, "S": 0.30, "C": 0.35}
PEAK = 0.89   # -1 dBFS


def pick_sources(n):
    """Per class, the n original train recordings the audio model rates highest."""
    ck = torch.load(AUDIO_CKPT, map_location="cpu")
    model = AudioCNN()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with open(CSV) as fh:
        originals = [r for r in csv.DictReader(fh)
                     if r["split"] == "train" and Path(r["path"]).stem.count("-") == 1]
    scored = {c: [] for c in CLASSES}
    torch.manual_seed(0)
    with torch.no_grad():
        for r in originals:
            label = int(r["label"])
            probs = torch.softmax(model(features(r["path"], ck["pad"], ck["norm"]).unsqueeze(0)), 1)[0]
            scored[CLASSES[label]].append((probs[label].item(), r["path"]))
    # Ties at full confidence are common; sorting on the path as well keeps
    # the pick the same on every run.
    return {c: sorted(v, key=lambda t: (-t[0], t[1]))[:n] for c, v in scored.items()}


def clean(path, token):
    """Mono, hit at the start, tail capped and faded, peak-normalised."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    assert sr == SR, f"{path}: {sr} Hz, expected an original at {SR}"
    a = np.abs(x)
    start = max(0, int(np.argmax(a > 0.1 * a.max())) - int(0.005 * sr))   # 5 ms before the hit
    x = x[start:start + int(MAX_LEN[token] * sr)]
    fade = min(len(x), int(0.010 * sr))
    x[-fade:] *= np.linspace(1, 0, fade)          # no click where the sound is cut
    return x / np.abs(x).max() * PEAK


def render(spec, cfg, takes, bars=8, gain=None):
    """Eight bars of one expression, variations rotating as in the app."""
    gain = {**cfg["gain"], **(gain or {})}   # the genre's balance over the shared one
    step = 60 / spec["bpm"] / spec.get("steps_per_beat", 2)
    n_steps = len(spec["A"])
    out = np.zeros(int(bars * n_steps * step * SR) + SR)
    turn = {t: 0 for t in NAME}   # round-robin over the takes of each sound
    for bar in range(bars):
        variation = cfg["order"][(bar // cfg["bars_per_variation"]) % len(cfg["order"])]
        for i, hit in enumerate(spec[variation]):
            at = int((bar * n_steps + i) * step * SR)
            for token in hit.replace("-", ""):
                s = takes[token][turn[token] % len(takes[token])] * gain[token]
                turn[token] += 1
                out[at:at + len(s)] += s
    out = out[:int(bars * n_steps * step * SR)]
    return out / max(1.0, np.abs(out).max() / PEAK)


def main():
    cfg = json.loads(PATTERNS.read_text())
    SAMPLES.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    takes = {}
    for cls, picks in sorted(pick_sources(cfg["samples_per_sound"]).items()):
        token = SOUND[cls]
        takes[token] = []
        for k, (conf, path) in enumerate(picks, 1):
            x = clean(path, token)
            takes[token].append(x)
            out = SAMPLES / f"{NAME[token]}_{k}.wav"
            sf.write(out, x, SR, subtype="PCM_16")
            print(f"{out.name:12} <- {parse_name(path)[1]:12} (model yakin {conf:.3f})  {len(x) / SR:.2f} s")
    for genre, g in cfg["genres"].items():
        for expr, spec in g["expressions"].items():
            out = render(spec, cfg, takes, gain=g.get("gain"))
            sf.write(PREVIEW / f"{genre}-{expr}.wav", out, SR, subtype="PCM_16")
            used = sorted({t for v in "ABC" for hit in spec[v] for t in hit.replace("-", "")})
            print(f"preview {genre:8} {expr:8} {spec['bpm']:3} BPM  bunyi {''.join(used):4}  {len(out) / SR:.1f} s")


if __name__ == "__main__":
    main()
