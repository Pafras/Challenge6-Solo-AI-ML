import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from show_batch import FER2013, build_transform
from train import build_model, evaluate

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--split", default="valid")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(ckpt["arch"], n_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    tf = build_transform(ckpt["image_size"], ckpt["channels"])
    loader = DataLoader (FER2013(args.split, transform=tf), batch_size=64, shuffle=False)

    loss, acc = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    print(f"Checkpoint : {ckpt['arch']}, epoch {ckpt['epoch']}, "
           f"Valid Waktu Disimpan {ckpt['val_acc']:.3f}")
    print(f"{args.split:10} : loss {loss:.4f} acc {acc:.3f}")