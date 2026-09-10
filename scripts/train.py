import argparse
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from torchvision import models
from pathlib import Path

from show_batch import CLASSES, FER2013, build_transform


class TinyCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 12 * 12, n_classes),)

    def forward(self, x):
        return self.net(x)


class DeepCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = total_n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n += len(labels)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total_n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n += len(labels)
    return total_loss / total_n, total_correct / total_n


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="tinycnn")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--save", default=None)
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd"])
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    return p.parse_args()
    


# What each architecture needs as input. The pretrained ones were trained on
# 224x224 RGB ImageNet photos, so the data is resized and repeated to match.
ARCH = {
    "tinycnn":   {"size": 48,  "channels": 1},
    "deepcnn":   {"size": 48,  "channels": 1},
    "resnet18":  {"size": 224, "channels": 3},
    "mobilenet": {"size": 224, "channels": 3},
}


def build_model(arch, n_classes=len(CLASSES)):
    if arch == "tinycnn":
        return TinyCNN(n_classes)
    if arch == "deepcnn":
        return DeepCNN(n_classes)
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, n_classes)   # 1000 ImageNet classes -> 4
        return m
    if arch == "mobilenet":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
        return m
    raise ValueError(f"Unknown architecture: {arch}")


def build_optimizer(name, params, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device {device} . arch {args.arch} . lr {args.lr} ."
          f"batch {args.batch_size} . epochs {args.epochs} . seed {args.seed}")

    # Built before the loaders: build_model rejects an unknown arch, so the
    # ARCH lookup below never sees a bad key.
    model = build_model(args.arch).to(device)

    spec =  ARCH[args.arch]
    train_tf = build_transform(image_size=spec["size"], channels=spec["channels"], augment=args.augment)
    valid_tf = build_transform(spec["size"], spec["channels"])
    train_loader = DataLoader(FER2013("train", transform=train_tf), batch_size=args.batch_size, shuffle=True) 
    valid_loader = DataLoader(FER2013("valid", transform=valid_tf), batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)

    # Cosine decays lr from its starting value to ~0 over the run: large steps
    # early to learn fast, small steps late to settle instead of wobbling.
    # T_max counts scheduler steps, and step() runs once per epoch below.
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion=criterion, optimizer=optimizer, device=device)
        va_loss, va_acc = evaluate(model, valid_loader, criterion=criterion, device=device)
        if va_acc > best:
            best = va_acc
            if args.save:
                Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "arch" : args.arch,
                    "classes" : CLASSES,
                    "image_size" : spec["size"],
                    "channels": spec["channels"],
                    "epoch" : epoch,
                    "val_acc" : va_acc,
                    "state_dict" : model.state_dict(),
                }, args.save)
                print(f"  -> Disimpan (epoch {epoch}, valid {va_acc:.3f})")
        print(f"Epoch {epoch:2} lr {lr_now:.1e}  train {tr_loss:.4f} / {tr_acc:.3f}   "
              f"valid {va_loss:.4f} / {va_acc:.3f}")
        if scheduler:
            scheduler.step()   # after the epoch's training, never inside run_epoch

    n_par = sum(q.numel() for q in model.parameters())
    aug = "flip+rot+jitter" if args.augment else "Tanpa"
    opt = f"{args.optimizer} wd{args.weight_decay}" if args.weight_decay else args.optimizer
    print(f"| ? | {args.arch} ({n_par} par) | {spec['size']} | {args.lr} | {opt} | {args.scheduler} | "
          f"{args.batch_size} | {aug} | Tanpa | {args.epochs} | {best:.3f} | |")