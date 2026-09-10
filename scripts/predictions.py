#!/usr/bin/env python3
"""Expected output against result output, for a trained checkpoint.

    python scripts/predictions.py models/mobilenet-cw.pt              # validation
    python scripts/predictions.py models/mobilenet-cw.pt --split test # Day 8, once

For every image: the label it carries (expected) and what the model said
(result). Prints one worked example with its loss, the confusion matrix, and
writes two pictures: a grid of right and wrong predictions, and the matrix.

The test split is opened once, on Day 8. Everything before that uses valid.
"""
import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from show_batch import FER2013, build_transform
from train import build_model

SURFACE, INK, INK_2, MUTED, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
# Sequential blue ramp from the reference palette: light means few, dark means many.
BLUES = LinearSegmentedColormap.from_list(
    "blues", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--split", default="valid")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    classes = ckpt["classes"]
    model = build_model(ckpt["arch"], n_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()

    ds = FER2013(args.split, transform=build_transform(ckpt["image_size"], ckpt["channels"]))
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(F.softmax(model(x.to(device)), dim=1).cpu())
            labels.append(y)
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    preds = probs.argmax(1)
    n = len(classes)

    # One worked example: the expected vector, the result vector, and the loss
    # that measures the distance between them.
    i = int(np.where(preds != labels)[0][0])
    expected = [1 if k == labels[i] else 0 for k in range(n)]
    print(f"contoh foto #{i}  ({ds.rows[i]['path']})")
    print(f"  expected : {expected}   = {classes[labels[i]]}")
    print(f"  result   : {[round(float(v), 2) for v in probs[i]]}   = tebakan {classes[preds[i]]}")
    print(f"  loss     : -ln({probs[i][labels[i]]:.2f}) = {-math.log(probs[i][labels[i]]):.2f}"
          "   (peluang yang dikasih ke jawaban benar)")

    cm = np.zeros((n, n), dtype=int)
    for t, r in zip(labels, preds):
        cm[t, r] += 1
    w = max(len(c) for c in classes)
    print(f"\nconfusion matrix ({args.split}) — baris = expected, kolom = result")
    print(" " * (w + 2) + "".join(f"{c:>10}" for c in classes) + "     benar")
    for t in range(n):
        print(f"{classes[t]:>{w}}  " + "".join(f"{cm[t, r]:>10}" for r in range(n))
              + f"     {cm[t, t] / cm[t].sum():.3f}")
    print(f"\ntotal benar {np.trace(cm)} / {cm.sum()} = {np.trace(cm) / cm.sum():.3f}")
    worst = max(((t, r) for t in range(n) for r in range(n) if t != r), key=lambda k: cm[k])
    print(f"paling sering ketuker: {classes[worst[0]]} ditebak {classes[worst[1]]} ({cm[worst]} foto)")

    Path("docs").mkdir(exist_ok=True)
    grid(ds, probs, labels, preds, classes, f"docs/predictions-{args.split}.png")
    matrix(cm, classes, f"docs/confusion-{args.split}.png", args.split)
    print(f"\nwrote docs/predictions-{args.split}.png, docs/confusion-{args.split}.png")


def grid(ds, probs, labels, preds, classes, out):
    """Eight right and eight wrong, drawn from a fixed shuffle so reruns match."""
    idx = list(range(len(labels)))
    random.Random(0).shuffle(idx)
    rows = [("Benar", [i for i in idx if preds[i] == labels[i]][:8]),
            ("Salah", [i for i in idx if preds[i] != labels[i]][:8])]
    fig, axes = plt.subplots(2, 8, figsize=(13, 4.4), facecolor=SURFACE)
    for r, (_, picks) in enumerate(rows):
        for c, i in enumerate(picks):
            ax = axes[r, c]
            ok = preds[i] == labels[i]
            ax.imshow(Image.open(ds.rows[i]["path"]), cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GOOD if ok else CRITICAL); s.set_linewidth(2.5)
            ax.set_title(f"{'✓' if ok else '✗'} harusnya {classes[labels[i]]}\n"
                         f"hasil {classes[preds[i]]} {probs[i][preds[i]]:.0%}",
                         fontsize=8.5, color=INK, loc="left")
    fig.text(0.01, 0.985, "Atas: tebakan benar (hijau ✓). Bawah: tebakan salah (merah ✗). "
             "Persen = seberapa yakin model sama tebakannya.",
             ha="left", va="top", fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=100, facecolor=SURFACE)
    plt.close(fig)


def matrix(cm, classes, out, split):
    n = len(classes)
    share = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=SURFACE)
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    for t in range(n):
        for r in range(n):
            dark = share[t, r] > 0.55
            ax.text(r, t, f"{cm[t, r]}\n{share[t, r]:.0%}", ha="center", va="center",
                    fontsize=10, color=SURFACE if dark else INK,
                    fontweight="semibold" if t == r else "normal")
    ax.set_xticks(range(n), classes, color=INK_2)
    ax.set_yticks(range(n), classes, color=INK_2)
    ax.set_xlabel("result — tebakan model", color=INK_2, labelpad=10)
    ax.set_ylabel("expected — label sebenarnya", color=INK_2, labelpad=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.set_title(f"Confusion matrix · {split}\ndiagonal = benar, sisanya = ke mana ia ketuker",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
