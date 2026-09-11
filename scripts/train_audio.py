#!/usr/bin/env python3
"""Train the beatbox sound classifier (clap / hihats / kick / snare).

    python scripts/train_audio.py --overfit          # 1 batch, must reach ~0 loss
    python scripts/train_audio.py --save models/audio-cnn.pt

The epoch loop, evaluation and optimizers are train.py's, so a number here
means the same thing as a number in the face table.

Two validation sets are measured every epoch. The bucket's own valid split
saturated at run A1 (3 of 1,044 wrong), so the best epoch is chosen on
avp-valid: 14 people the bucket never recorded, on a MacBook mic. That is
also what the chart shows as "valid".
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

AVP_CSV = "data/splits/avp_3class.csv"


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
def recall_by(model, ds, groups, device):
    """Share of clips predicted correctly, per group (class name or AVP label)."""
    model.eval()
    pred = model(ds.x.to(device)).argmax(1).cpu()
    hit, tot = Counter(), Counter()
    for g, p, t in zip(groups, pred.tolist(), ds.y.tolist()):
        tot[g] += 1
        hit[g] += p == t
    return {g: hit[g] / tot[g] for g in sorted(tot)}


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
    p.add_argument("--pad", default="zero", choices=["zero", "noise"])
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
          f"epochs {args.epochs} · pad {args.pad} · seed {args.seed}")

    # Every split goes through the same padding: it is part of the pipeline,
    # not an augmentation, and the app would have to reproduce it too.
    train_ds = BeatboxAudio("train", pad=args.pad)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    if args.overfit:
        overfit_one_batch(model, train_loader, device)
        raise SystemExit

    bucket_ds = BeatboxAudio("valid", pad=args.pad)
    avp_ds = BeatboxAudio("avp-valid", AVP_CSV, pad=args.pad)
    bucket_loader = DataLoader(bucket_ds, batch_size=args.batch_size, shuffle=False)
    avp_loader = DataLoader(avp_ds, batch_size=args.batch_size, shuffle=False)

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
        bu_loss, bu_acc = evaluate(model, bucket_loader, criterion, device)
        av_loss, av_acc = evaluate(model, avp_loader, criterion, device)
        if av_acc > best:
            best, best_epoch, best_bucket = av_acc, epoch, bu_acc
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch:2} lr {lr_now:.1e}  train {tr_loss:.4f} / {tr_acc:.3f}   "
              f"bucket {bu_loss:.4f} / {bu_acc:.3f}   avp {av_loss:.4f} / {av_acc:.3f}")
        history.append({"epoch": epoch, "lr": lr_now, "train_loss": tr_loss, "train_acc": tr_acc,
                        "valid_loss": av_loss, "valid_acc": av_acc,
                        "bucket_loss": bu_loss, "bucket_acc": bu_acc})
        if scheduler:
            scheduler.step()
    seconds = time.time() - start

    model.load_state_dict(best_state)
    bucket_recall = recall_by(model, bucket_ds, [CLASSES[y] for y in bucket_ds.y.tolist()], device)
    avp_recall = recall_by(model, avp_ds, [r["avp_label"] for r in avp_ds.rows], device)
    print(f"terbaik epoch {best_epoch}: avp {best:.3f} · bucket {best_bucket:.3f} · {seconds:.0f} detik")
    print("  bucket recall: " + " · ".join(f"{c} {r:.3f}" for c, r in bucket_recall.items()))
    print("  avp recall:    " + " · ".join(f"{c} {r:.3f}" for c, r in avp_recall.items()))

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": "audiocnn", "classes": CLASSES, "pad": args.pad, "epoch": best_epoch,
                    "avp_acc": best, "bucket_acc": best_bucket, "state_dict": best_state}, args.save)
        print(f"disimpan: {args.save}")

    opt = f"{args.optimizer} wd{args.weight_decay}" if args.weight_decay else args.optimizer
    cw = "balanced" if args.class_weight else "tanpa"
    print(f"| ? | AudioCNN ({n_par} par) | log-mel 64x44, pad {args.pad} | {args.lr} | {opt} | "
          f"{args.scheduler} | {args.batch_size} | tanpa | {cw} | {args.epochs} | "
          f"{best:.3f} / {best_bucket:.3f} | |")

    name = Path(args.save).stem if args.save else "audiocnn"
    curves = Path("docs/curves")
    curves.mkdir(parents=True, exist_ok=True)
    record = {
        "name": name,
        "config": vars(args),
        "summary": (f"AudioCNN · pad {args.pad} · lr {args.lr} · {opt} · scheduler {args.scheduler} · "
                    f"class weight {cw} · {args.epochs} epoch · valid = avp-valid, terbaik {best:.3f} "
                    f"(bucket {best_bucket:.3f})"),
        "bucket_recall": bucket_recall,
        "avp_recall": avp_recall,
        "epochs": history,
    }
    (curves / f"{name}.json").write_text(json.dumps(record, indent=2))
    plot_run(record, curves / f"{name}.png")
    print(f"grafik: {curves / name}.png")
