#!/usr/bin/env python3
"""Train the beatbox sound classifier (clap / hihats / kick / snare).

    python scripts/train_audio.py --overfit          # 1 batch, must reach ~0 loss
    python scripts/train_audio.py --save models/audio-cnn.pt

The epoch loop, evaluation and optimizers are train.py's, so a number here
means the same thing as a number in the face table.
"""
import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from audio_features import BeatboxAudio
from audio_labels import CLASSES
from plot_runs import plot_run
from train import build_optimizer, evaluate, run_epoch


class AudioCNN(nn.Module):
    """Three conv blocks on a 64 x 44 log-mel, then average over what is left.

    64x44 -> 32x22 -> 16x11 -> 8x5, and the average makes the head
    independent of the exact input size."""

    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()

        def block(c_in, c_out):
            return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                    nn.ReLU(), nn.MaxPool2d(2)]

        self.net = nn.Sequential(
            *block(1, 16), *block(16, 32), *block(32, 64),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.net(x)


def overfit_one_batch(model, loader, device, steps=200):
    """A model that cannot memorise 32 clips has a bug, not a tuning problem."""
    x, y = next(iter(loader))
    print("bentuk batch:", tuple(x.shape), " label:", dict(Counter(CLASSES[i] for i in y.tolist())))
    x, y = x.to(device), y.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(1, steps + 1):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 20 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"step {step:3}  loss {loss.item():.4f}  acc {acc:.2f}")
    assert loss.item() < 0.05, f"loss still {loss.item():.3f} after {steps} steps"
    print("overfit 1 batch: ok")


@torch.no_grad()
def per_class_recall(model, loader, device):
    model.eval()
    hit, tot = Counter(), Counter()
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for p, t in zip(pred.tolist(), y.tolist()):
            tot[t] += 1
            hit[t] += p == t
    return {CLASSES[c]: hit[c] / tot[c] for c in range(len(CLASSES))}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd"])
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    p.add_argument("--class-weight", action="store_true")
    p.add_argument("--save", default=None)
    p.add_argument("--overfit", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AudioCNN().to(device)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"device {device} · AudioCNN ({n_par} par) · lr {args.lr} · batch {args.batch_size} · "
          f"epochs {args.epochs} · seed {args.seed}")

    train_ds = BeatboxAudio("train")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    if args.overfit:
        overfit_one_batch(model, train_loader, device)
        raise SystemExit

    valid_loader = DataLoader(BeatboxAudio("valid"), batch_size=args.batch_size, shuffle=False)

    weight = None
    if args.class_weight:
        counts = Counter(train_ds.y.tolist())
        weights = torch.tensor([len(train_ds) / (len(CLASSES) * counts[i]) for i in range(len(CLASSES))])
        print("class weight:", {c: round(float(w), 2) for c, w in zip(CLASSES, weights)})
        weight = weights.to(device)
    # Class weights reshape the training loss only; validation keeps the plain one.
    train_criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best, best_state, history = 0.0, None, []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = run_epoch(model, train_loader, train_criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, valid_loader, criterion, device)
        if va_acc > best:
            best, best_epoch = va_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch:2} lr {lr_now:.1e}  train {tr_loss:.4f} / {tr_acc:.3f}   "
              f"valid {va_loss:.4f} / {va_acc:.3f}")
        history.append({"epoch": epoch, "lr": lr_now, "train_loss": tr_loss, "train_acc": tr_acc,
                        "valid_loss": va_loss, "valid_acc": va_acc})
        if scheduler:
            scheduler.step()
    seconds = time.time() - start

    model.load_state_dict(best_state)
    recall = per_class_recall(model, valid_loader, device)
    print(f"terbaik epoch {best_epoch}: valid {best:.3f} · " +
          " · ".join(f"{c} {r:.3f}" for c, r in recall.items()) + f" · {seconds:.0f} detik")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": "audiocnn", "classes": CLASSES, "epoch": best_epoch,
                    "val_acc": best, "state_dict": best_state}, args.save)
        print(f"disimpan: {args.save}")

    opt = f"{args.optimizer} wd{args.weight_decay}" if args.weight_decay else args.optimizer
    cw = "balanced" if args.class_weight else "tanpa"
    print(f"| ? | AudioCNN ({n_par} par) | log-mel 64x44 | {args.lr} | {opt} | {args.scheduler} | "
          f"{args.batch_size} | tanpa | {cw} | {args.epochs} | {best:.3f} | |")

    name = Path(args.save).stem if args.save else "audiocnn"
    curves = Path("docs/curves")
    curves.mkdir(parents=True, exist_ok=True)
    record = {
        "name": name,
        "config": vars(args),
        "summary": (f"AudioCNN · lr {args.lr} · {opt} · scheduler {args.scheduler} · "
                    f"class weight {cw} · {args.epochs} epoch · valid terbaik {best:.3f}"),
        "per_class_recall": recall,
        "epochs": history,
    }
    (curves / f"{name}.json").write_text(json.dumps(record, indent=2))
    plot_run(record, curves / f"{name}.png")
    print(f"grafik: {curves / name}.png")
