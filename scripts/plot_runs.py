#!/usr/bin/env python3
"""Training curves as pictures.

    python scripts/plot_runs.py docs/curves/mobilenet-adamw.json
    python scripts/plot_runs.py docs/curves/a.json docs/curves/b.json --out docs/curves/compare.png

train.py writes docs/curves/<name>.json after every run and draws its chart
with plot_run(). Given several files, this script overlays their validation
curves instead, one line per run.

Loss, accuracy and learning rate each get their own panel. Never one chart
with two y-axes: the scales differ, and aligning them arbitrarily invents a
relationship that is not in the data.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference palette (dataviz skill, light mode). Series slots are used in
# this fixed order and validated as a set; text never takes a series colour.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GAP = "#f0efec"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TRAIN, VALID = SERIES[0], SERIES[1]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})

LINE = dict(linewidth=2, marker="o", markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1.5)


def load(path):
    return json.loads(Path(path).read_text())


def _style(ax, ylabel):
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


def _end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK_2)


def _mark(ax, x, y, text, above=True):
    ax.plot([x], [y], marker="o", markersize=9, markerfacecolor="none",
            markeredgecolor=INK, markeredgewidth=1.4, linestyle="none")
    ax.annotate(text, (x, y), xytext=(0, 10 if above else -14),
                textcoords="offset points", ha="center", fontsize=9, color=INK)


def plot_run(history, out):
    """One run: train vs valid, loss and accuracy (and lr when it moves)."""
    ep = [e["epoch"] for e in history["epochs"]]
    get = lambda k: [e[k] for e in history["epochs"]]
    lrs = get("lr") if "lr" in history["epochs"][0] else None
    show_lr = lrs is not None and len(set(lrs)) > 1

    n = 3 if show_lr else 2
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2))
    last = ep[-1]

    ax = axes[0]
    ax.plot(ep, get("train_loss"), color=TRAIN, label="train", **LINE)
    ax.plot(ep, get("valid_loss"), color=VALID, label="valid", **LINE)
    vl = get("valid_loss")
    i = vl.index(min(vl))
    _mark(ax, ep[i], vl[i], f"terendah {vl[i]:.3f}", above=False)
    _end_label(ax, last, get("train_loss")[-1], "train")
    _end_label(ax, last, vl[-1], "valid")
    ax.set_title("Loss  (lebih rendah = lebih baik)", loc="left")
    _style(ax, "loss")

    ax = axes[1]
    tr, va = get("train_acc"), get("valid_acc")
    # The shaded band is the train-valid gap: how far the model does better on
    # images it has seen than on ones it has not. A widening band is memorising.
    ax.fill_between(ep, va, tr, where=[t >= v for t, v in zip(tr, va)],
                    color=GAP, linewidth=0, label="gap train–valid")
    ax.plot(ep, tr, color=TRAIN, label="train", **LINE)
    ax.plot(ep, va, color=VALID, label="valid", **LINE)
    j = va.index(max(va))
    _mark(ax, ep[j], va[j], f"terbaik {va[j]:.3f}")
    _end_label(ax, last, tr[-1], "train")
    _end_label(ax, last, va[-1], "valid")
    ax.set_title("Akurasi  (lebih tinggi = lebih baik)", loc="left")
    _style(ax, "akurasi")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)

    if show_lr:
        ax = axes[2]
        ax.plot(ep, lrs, color=SERIES[2], **LINE)
        ax.set_title("Learning rate", loc="left")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        _style(ax, "lr")

    fig.suptitle(history["name"], x=0.01, y=0.985, ha="left", fontsize=13,
                 fontweight="semibold", color=INK)
    fig.text(0.01, 0.905, history.get("summary", ""), ha="left",
             fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _label_ends(ax, ends):
    """Name each line at its last point, pushed apart vertically when two
    finishes are close (0.810 vs 0.797) so they do not overprint.

    Only lines that reach the right edge get a direct label. A run that stops
    early ends in the middle of the plot, where its label would sit on top of
    the runs that continue - hiding data is worse than an unlabelled line.
    Those runs are identified by the legend, which is always drawn."""
    last_x = max(e[0] for e in ends)
    lo, hi = ax.get_ylim()
    gap = 0.055 * (hi - lo)
    prev = None
    for e in sorted((e for e in ends if e[0] == last_x), key=lambda e: e[1]):
        y = e[1] if prev is None else max(e[1], prev + gap)
        prev = y
        ax.annotate(e[2], (last_x, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    left, right = ax.get_xlim()
    ax.set_xlim(left, right + 0.22 * (right - left))
    # The extra room is for labels, not epochs: keep ticks inside the data.
    ax.set_xticks([x for x in ax.get_xticks() if 1 <= x <= last_x])


def plot_compare(histories, out):
    """Several runs: validation loss and accuracy, one line per run."""
    if len(histories) > len(SERIES):
        raise SystemExit(f"max {len(SERIES)} run per grafik — pecah jadi beberapa")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for key, ax, title, best in [
        ("valid_loss", axes[0], "Loss valid  (lebih rendah = lebih baik)", min),
        ("valid_acc", axes[1], "Akurasi valid  (lebih tinggi = lebih baik)", max),
    ]:
        ends = []
        for color, h in zip(SERIES, histories):
            ep = [e["epoch"] for e in h["epochs"]]
            ys = [e[key] for e in h["epochs"]]
            ax.plot(ep, ys, color=color, label=h["name"], **LINE)
            k = ys.index(best(ys))
            ax.plot([ep[k]], [ys[k]], marker="o", markersize=9, markerfacecolor="none",
                    markeredgecolor=INK, markeredgewidth=1.2, linestyle="none")
            ends.append([ep[-1], ys[-1], h["name"]])
        ax.set_title(title, loc="left")
        _style(ax, {"valid_loss": "loss", "valid_acc": "akurasi"}[key])
        _label_ends(ax, ends)
    axes[1].legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.text(0.01, 0.96, "Lingkaran hitam = titik terbaik tiap run", ha="left",
             fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs = [load(r) for r in args.runs]
    if len(runs) == 1:
        out = args.out or str(Path(args.runs[0]).with_suffix(".png"))
        plot_run(runs[0], out)
    else:
        out = args.out or "docs/curves/compare.png"
        plot_compare(runs, out)
    print(f"wrote {out}")
