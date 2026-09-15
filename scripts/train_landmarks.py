#!/usr/bin/env python3
"""Step 2 of the landmark run: a face-shape classifier, no pixels at all.

    python scripts/train_landmarks.py --check                # overfit one batch
    python scripts/train_landmarks.py --save models/landmark-mlp.pt

Input: the 76 Vision points from extract_landmarks.py, 152 numbers per face.
Model: the paper's "optimized MLP" (Juarez-Jimenez et al., 2025),
512-256-128 with BatchNorm, GELU, dropout 0.3 and a skip connection
around every block.

Same training data as the shipped CNN (Fine-tune bersih): FER2013 train
plus ana, bintang, cile and firda repeated --repeat times. Same valid sets:
FER valid, and erin, who never trained. Same class weights and label
smoothing, so the two models' probabilities are on a similar scale when
step 3 averages them. Faces Vision missed (~2%) are left out of both
training and scoring here; step 3 falls back to the CNN for them.

Inputs are standardised with train's mean and std per coordinate. Both
go into the checkpoint: the app must apply exactly these, or the MLP sees
numbers it was never trained on.
"""
import argparse
import copy
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fine_tune import per_class_recall
from plot_runs import plot_run
from train import build_optimizer, evaluate, run_epoch

CLASSES = ["angry", "happy", "neutral", "surprise", "sad"]
FER = "data/landmarks/fer2013_5class.npz"
OWN = "data/landmarks/own_faces_semua.npz"
OWN_CSV = "data/splits/own_faces_bersih.csv"


class Block(nn.Module):
    def __init__(self, n_in, n_out, dropout):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n_in, n_out), nn.BatchNorm1d(n_out), nn.GELU(), nn.Dropout(dropout))
        # Sizes change at every block, so the skip needs its own projection.
        self.skip = nn.Linear(n_in, n_out, bias=False)

    def forward(self, x):
        return self.f(x) + self.skip(x)


class LandmarkMLP(nn.Module):
    def __init__(self, n_in=152, n_classes=len(CLASSES), dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(Block(n_in, 512, dropout), Block(512, 256, dropout),
                                 Block(256, 128, dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x)


def load(npz, split, keep_paths=None):
    """(x, y) of the faces Vision found in one split, optionally only these paths."""
    d = np.load(npz)
    keep = d["found"] & (d["split"] == split)
    if keep_paths is not None:
        keep &= np.isin(d["path"], list(keep_paths))
    return torch.from_numpy(d["x"][keep]), torch.from_numpy(d["label"][keep]).long()


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check", action="store_true", help="overfit one batch, then stop")
    p.add_argument("--save", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cpu"   # 152 numbers per face: MPS start-up costs more than it saves

    with open(OWN_CSV) as fh:
        bersih = {r["path"] for r in csv.DictReader(fh)}
    fer_x, fer_y = load(FER, "train")
    own_x, own_y = load(OWN, "train", bersih)
    train_x = torch.cat([fer_x] + [own_x] * args.repeat)
    train_y = torch.cat([fer_y] + [own_y] * args.repeat)
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    norm = lambda x: (x - mean) / std
    fv_x, fv_y = load(FER, "valid")
    ov_x, ov_y = load(OWN, "valid", bersih)
    print(f"train: FER {len(fer_x)} + own {len(own_x)} x{args.repeat} · "
          f"FER valid {len(fv_x)} · own valid (erin) {len(ov_x)}")

    model = LandmarkMLP(dropout=args.dropout).to(device)
    print(f"parameter: {sum(p.numel() for p in model.parameters()):,}")
    counts = Counter(fer_y.tolist())
    weight = torch.tensor([len(fer_y) / (len(CLASSES) * counts[i]) for i in range(len(CLASSES))])
    train_criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer("adamw", model.parameters(), args.lr, args.weight_decay)

    if args.check:
        # The loop is only trustworthy if it can memorise 64 faces. Without
        # label smoothing or dropout the loss should head for zero.
        xb, yb = norm(train_x[:64]), train_y[:64]
        model = LandmarkMLP(dropout=0.0)
        opt = build_optimizer("adamw", model.parameters(), 1e-3, 0.0)
        for step in range(300):
            loss = nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0:
                print(f"step {step:3} loss {loss.item():.4f}")
        print(f"step 300 loss {loss.item():.4f}")
        assert loss.item() < 0.05, "cannot overfit one batch: the loop is broken"
        print("ok")
        raise SystemExit

    loader = lambda x, y, shuffle: DataLoader(TensorDataset(norm(x), y), args.batch_size, shuffle=shuffle)
    train_loader = loader(train_x, train_y, True)
    fer_valid, own_valid = loader(fv_x, fv_y, False), loader(ov_x, ov_y, False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best, best_state, best_epoch, history = 0.0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = run_epoch(model, train_loader, train_criterion, optimizer, device)
        fv_loss, fv_acc = evaluate(model, fer_valid, criterion, device)
        ov_loss, ov_acc = evaluate(model, own_valid, criterion, device)
        # Picked on FER valid, like every face run before the fine-tunes.
        # Erin is 300 frames of one person: too few to choose an epoch on.
        if fv_acc > best:
            best, best_epoch, best_state = fv_acc, epoch, copy.deepcopy(model.state_dict())
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2} lr {lr_now:.1e}  train {tr_loss:.4f} / {tr_acc:.3f}   "
                  f"FER valid {fv_loss:.4f} / {fv_acc:.3f}   erin {ov_loss:.4f} / {ov_acc:.3f}")
        history.append({"epoch": epoch, "lr": lr_now, "train_loss": tr_loss, "train_acc": tr_acc,
                        "valid_loss": fv_loss, "valid_acc": fv_acc, "own_loss": ov_loss, "own_acc": ov_acc})
        scheduler.step()

    model.load_state_dict(best_state)
    _, own_best = evaluate(model, own_valid, criterion, device)
    print(f"\nterbaik epoch {best_epoch}: FER valid {best:.3f} · erin {own_best:.3f}")
    for name, dl in [("FER valid", fer_valid), ("erin", own_valid)]:
        recall = per_class_recall(model, dl, CLASSES, device)
        print(f"  recall {name:9}: " + " · ".join(f"{c} {r:.2f}" for c, r in recall.items()))

    name = Path(args.save).stem if args.save else "landmark-mlp"
    if args.save:
        torch.save({"arch": "landmark-mlp", "classes": CLASSES, "dropout": args.dropout,
                    "mean": mean, "std": std, "epoch": best_epoch, "val_acc": best,
                    "own_val_acc": own_best, "state_dict": best_state}, args.save)
        print(f"disimpan: {args.save}")
    curves = Path("docs/curves")
    record = {"name": name, "config": vars(args), "epochs": history,
              "summary": (f"Landmark MLP · 76 titik Vision · lr {args.lr} · {args.epochs} epoch · "
                          f"terbaik epoch {best_epoch}: FER valid {best:.3f}, erin {own_best:.3f}")}
    (curves / f"{name}.json").write_text(json.dumps(record, indent=2))
    plot_run(record, curves / f"{name}.png")
    print(f"grafik: {curves / name}.png")
