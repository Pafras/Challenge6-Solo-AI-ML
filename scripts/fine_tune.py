#!/usr/bin/env python3
"""Fine-tune the +Sad face model on recorded faces, with FER2013 kept in.

    python scripts/fine_tune.py --save models/mobilenet-finetune.pt

Starts from models/mobilenet-ls-sad.pt, not from ImageNet: the model
already knows FER2013's faces, and the aim is to adjust that to faces like
ours, not relearn it. Hence a learning rate a tenth of the original.

Training data is FER2013 train plus the recorded train people, repeated
--repeat times. A few hundred recorded frames against 20,550 FER images
would otherwise be noise. FER stays in so the model does not forget every
face that is not ours (catastrophic forgetting).

Measured every epoch, and once before the first:
    own valid   people who never trained. Did it learn faces like ours?
    FER valid   did it forget the rest?
The best epoch is picked on own valid. Epoch 0 is the untouched +Sad
model on the same frames, the honest "before".
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from plot_runs import plot_run
from show_batch import FER2013, build_transform, load_rows
from train import build_model, build_optimizer, evaluate, run_epoch


@torch.no_grad()
def per_class_recall(model, loader, classes, device):
    model.eval()
    hit, tot = Counter(), Counter()
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for p, t in zip(pred.tolist(), y.tolist()):
            tot[t] += 1
            hit[t] += p == t
    return {classes[c]: hit[c] / tot[c] for c in sorted(tot)}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="models/mobilenet-ls-sad.pt")
    p.add_argument("--own-csv", default="data/splits/own_faces.csv")
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    ck = torch.load(args.init, map_location="cpu")
    classes = ck["classes"]
    model = build_model(ck["arch"], n_classes=len(classes))
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    fer_csv = f"data/splits/fer2013_{len(classes)}class.csv"

    train_tf = build_transform(ck["image_size"], ck["channels"], augment=True)
    valid_tf = build_transform(ck["image_size"], ck["channels"])
    own_train = FER2013("train", train_tf, csv_path=args.own_csv)
    train_ds = ConcatDataset([FER2013("train", train_tf, csv_path=fer_csv)] + [own_train] * args.repeat)
    loader = lambda split, csv_path: DataLoader(FER2013(split, valid_tf, csv_path=csv_path),
                                                batch_size=args.batch_size, shuffle=False)
    own_valid, fer_valid = loader("valid", args.own_csv), loader("valid", fer_csv)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    people = {r["person"]: r["split"] for r in load_rows("train", args.own_csv) + load_rows("valid", args.own_csv)}
    print(f"device {device} · init {args.init} · lr {args.lr} · repeat {args.repeat} · epochs {args.epochs}")
    print(f"train: FER {len(train_ds) - len(own_train) * args.repeat} + own {len(own_train)} x{args.repeat} · "
          f"own valid: {[p for p, s in people.items() if s == 'valid']}")

    # Same balanced class weights as the +Sad run, from FER2013's counts, so
    # the loss means what it meant when the base model was trained.
    counts = Counter(int(r["label"]) for r in load_rows("train", fer_csv))
    total = sum(counts.values())
    weight = torch.tensor([total / (len(classes) * counts[i]) for i in range(len(classes))]).to(device)
    train_criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer("adamw", model.parameters(), args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ov_loss, ov_acc = evaluate(model, own_valid, criterion, device)
    fv_loss, fv_acc = evaluate(model, fer_valid, criterion, device)
    before = per_class_recall(model, own_valid, classes, device)
    print(f"Epoch  0 (sebelum)  own valid {ov_loss:.4f} / {ov_acc:.3f}   FER valid {fv_loss:.4f} / {fv_acc:.3f}")
    print("  own recall sebelum: " + " · ".join(f"{c} {r:.2f}" for c, r in before.items()))

    best, best_state, best_epoch = ov_acc, copy.deepcopy(model.state_dict()), 0
    history = []
    for epoch in range(1, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = run_epoch(model, train_loader, train_criterion, optimizer, device)
        ov_loss, ov_acc = evaluate(model, own_valid, criterion, device)
        fv_loss, fv_acc = evaluate(model, fer_valid, criterion, device)
        if ov_acc > best:
            best, best_epoch, best_state = ov_acc, epoch, copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch:2} lr {lr_now:.1e}  train {tr_loss:.4f} / {tr_acc:.3f}   "
              f"own valid {ov_loss:.4f} / {ov_acc:.3f}   FER valid {fv_loss:.4f} / {fv_acc:.3f}")
        history.append({"epoch": epoch, "lr": lr_now, "train_loss": tr_loss, "train_acc": tr_acc,
                        "valid_loss": ov_loss, "valid_acc": ov_acc, "fer_loss": fv_loss, "fer_acc": fv_acc})
        scheduler.step()

    model.load_state_dict(best_state)
    after = per_class_recall(model, own_valid, classes, device)
    _, fer_best = evaluate(model, fer_valid, criterion, device)
    print(f"\nterbaik epoch {best_epoch}: own valid {best:.3f} · FER valid {fer_best:.3f}")
    print("  own recall sesudah: " + " · ".join(f"{c} {r:.2f}" for c, r in after.items()))

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({**{k: ck[k] for k in ("arch", "image_size", "channels")}, "classes": classes,
                    "epoch": best_epoch, "val_acc": fer_best, "own_val_acc": best,
                    "fine_tuned_from": args.init, "state_dict": best_state}, args.save)
        print(f"disimpan: {args.save}")

    name = Path(args.save).stem if args.save else "finetune"
    curves = Path("docs/curves")
    curves.mkdir(parents=True, exist_ok=True)
    record = {
        "name": name,
        "config": vars(args),
        "summary": (f"fine-tune dari {Path(args.init).stem} · lr {args.lr} · repeat {args.repeat} · "
                    f"{args.epochs} epoch · valid = orang yang gak ikut dilatih, terbaik {best:.3f} "
                    f"(FER valid {fer_best:.3f})"),
        "own_recall_before": before,
        "own_recall_after": after,
        "epochs": history,
    }
    (curves / f"{name}.json").write_text(json.dumps(record, indent=2))
    plot_run(record, curves / f"{name}.png")
    print(f"grafik: {curves / name}.png")
