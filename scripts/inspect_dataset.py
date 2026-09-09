#!/usr/bin/env python3
"""Sanity-check a facial-expression dataset before trusting it.

Usage:  python scripts/inspect_dataset.py data/fer2013/train
        python scripts/inspect_dataset.py --selftest

Answers two questions: how lopsided are the classes, and can a human read
the labels? The second one is the eyeball test — open the PNG it writes and
try to label the faces yourself. Labels you can't read, a model can't either.
"""
import sys
from collections import Counter
from pathlib import Path


def inspect(root, out="docs/dataset-eyeball.png", per_class=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.datasets import ImageFolder

    ds = ImageFolder(root)
    counts = Counter(ds.targets)

    for i, name in enumerate(ds.classes):
        print(f"{name:12} {counts[i]:6}")
    ratio = max(counts.values()) / min(counts.values())
    print(f"\ntotal {len(ds)}, imbalance {ratio:.1f}x")
    if ratio > 3:
        print("-> lopsided. class weights or oversampling, or drop the small class.")
    if min(counts.values()) < 500:
        print("-> a class under 500 samples will be hard to learn.")

    rows = len(ds.classes)
    fig, axes = plt.subplots(rows, per_class, figsize=(per_class * 1.6, rows * 1.8))
    axes = axes.reshape(rows, per_class)
    for i, name in enumerate(ds.classes):
        picks = [j for j, t in enumerate(ds.targets) if t == i][:per_class]
        for ax, j in zip(axes[i], picks):
            ax.imshow(ds[j][0], cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
        axes[i][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=90)
    print(f"\nwrote {out} — open it and try to label the faces yourself.")


def selftest():
    """Build a tiny fake dataset, check the counts come back right."""
    import tempfile
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        for name, n in [("happy", 3), ("angry", 1)]:
            d = Path(tmp) / name
            d.mkdir()
            for k in range(n):
                Image.new("L", (8, 8), color=k * 40).save(d / f"{k}.png")

        from torchvision.datasets import ImageFolder
        ds = ImageFolder(tmp)
        counts = Counter(ds.targets)
        assert ds.classes == ["angry", "happy"], ds.classes
        assert counts[ds.class_to_idx["happy"]] == 3
        assert counts[ds.class_to_idx["angry"]] == 1
        inspect(tmp, out=str(Path(tmp) / "grid.png"), per_class=1)
        assert (Path(tmp) / "grid.png").exists()
    print("ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        sys.exit(__doc__)
