#!/usr/bin/env python3
"""Step 4 of the landmark run: a landmark box plot per emotion, after the paper.

    python scripts/plot_landmarks.py     # -> docs/landmark-boxplot.png

One panel per class, from FER2013 train faces Vision found points on. Per
point, the shaded box spans the middle half of all faces (25th to 75th
percentile) in x and in y, and the line joins the per-point medians into
a face. Every panel but neutral also draws neutral's median face in grey,
so what an expression moves is the gap between the two lines.

Units are pupil distances, centred between the pupils (extract_landmarks.py).
Vision's y points up, so the plot needs no flip. Outliers are left out:
76 points x 4,000 faces of them would bury the shape.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from plot_runs import INK, INK_2, MUTED, SERIES, SURFACE
from train_landmarks import CLASSES, FER

# Index of each region inside Vision's 76 allPoints, matched point for point
# on FER faces (identical on every face checked). Closed shapes repeat
# their first index so the outline meets itself.
REGIONS = {
    "contour": list(range(59, 76)),
    "left brow": [19, 18, 17, 16, 15, 14],
    "right brow": [25, 24, 23, 22, 21, 20],
    "nose crest": [46, 47, 48, 49, 58, 57],
    "nose": [47, 56, 54, 53, 52, 51, 50, 55],
    "left eye": [0, 4, 5, 1, 3, 2, 0],
    "right eye": [7, 11, 12, 8, 10, 9, 7],
    "outer lips": list(range(26, 40)) + [26],
    "inner lips": [42, 40, 43, 45, 41, 44, 42],
}
COLOR = SERIES[0]
ORDER = ["neutral", "happy", "surprise", "angry", "sad"]


def draw_face(ax, pts, color, width, z):
    for idx in REGIONS.values():
        ax.plot(pts[idx, 0], pts[idx, 1], color=color, linewidth=width, zorder=z,
                solid_capstyle="round")


if __name__ == "__main__":
    d = np.load(FER)
    keep = d["found"] & (d["split"] == "train")
    pts, labels = d["x"][keep].reshape(-1, 76, 2), d["label"][keep]
    median = {c: np.median(pts[labels == i], 0) for i, c in enumerate(CLASSES)}

    fig, axes = plt.subplots(2, 3, figsize=(12, 9.6))
    for ax, name in zip(axes.flat, ORDER):
        faces = pts[labels == CLASSES.index(name)]
        lo, hi = np.percentile(faces, 25, 0), np.percentile(faces, 75, 0)
        for (x0, y0), (x1, y1) in zip(lo, hi):
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=COLOR,
                                   alpha=0.18, edgecolor="none", zorder=1))
        if name != "neutral":
            draw_face(ax, median["neutral"], MUTED, 1.4, 2)
        draw_face(ax, median[name], COLOR, 2, 3)
        ax.set_title(f"{name}   n = {len(faces):,}".replace(",", "."), loc="left")
        ax.set_aspect("equal")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-2.35, 0.95)
        ax.axis("off")

    # Sixth cell: the key, in words, instead of a legend box squeezed into a face.
    key = axes.flat[5]
    key.axis("off")
    key.plot([0.02, 0.14], [0.86, 0.86], color=COLOR, linewidth=2, transform=key.transAxes)
    key.text(0.18, 0.86, "median emosi itu", va="center", color=INK, transform=key.transAxes)
    key.plot([0.02, 0.14], [0.76, 0.76], color=MUTED, linewidth=1.4, transform=key.transAxes)
    key.text(0.18, 0.76, "median neutral (pembanding)", va="center", color=INK, transform=key.transAxes)
    key.add_patch(Rectangle((0.02, 0.63), 0.12, 0.06, facecolor=COLOR, alpha=0.18,
                            edgecolor="none", transform=key.transAxes))
    key.text(0.18, 0.66, "kotak = 50% wajah di tengah\n(persentil 25–75, per titik)",
             va="center", color=INK, transform=key.transAxes)
    key.text(0.02, 0.44, "Satuan: jarak antar pupil,\ndipusatkan di tengah kedua pupil.\n\n"
             "Kotak besar = titik itu beda-beda\nantar wajah; kotak kecil = konsisten.",
             va="top", color=INK_2, fontsize=9.5, transform=key.transAxes, linespacing=1.5)

    fig.suptitle("Landmark box plot · 76 titik Vision · FER2013 train", x=0.02, y=0.985,
                 ha="left", fontsize=13, fontweight="semibold", color=INK)
    fig.text(0.02, 0.945, "Ekspresi = jarak antara garis biru dan abu-abu. "
             "Di mana dua garis nempel, titik itu gak bantu bedain emosi.",
             ha="left", fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("docs/landmark-boxplot.png", dpi=150, facecolor=SURFACE)
    print("disimpan: docs/landmark-boxplot.png")
