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
    # The checkpoint's own class list picks the split: a 5-class model scored
    # on the 4-class CSV would never see a sad face.
    n = len(ckpt["classes"])
    csv_path = f"data/splits/fer2013_{n}class.csv"
    loader = DataLoader (FER2013(args.split, transform=tf, csv_path=csv_path), batch_size=64, shuffle=False)

    loss, acc = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    print(f"Checkpoint : {ckpt['arch']}, epoch {ckpt['epoch']}, "
           f"Valid Waktu Disimpan {ckpt['val_acc']:.3f}")
    print(f"{args.split:10} : loss {loss:.4f} acc {acc:.3f}")

    # Total accuracy hides which emotions the model gets wrong. Per class,
    # a gain on a small class can be seen even when it costs a large one.
    model.eval()
    classes = ckpt["classes"]
    confusion = [[0] * n for _ in range(n)]
    with torch.no_grad():
        for images, labels in loader:
            pred = model(images.to(device)).argmax(1).cpu()
            for y, yhat in zip(labels.tolist(), pred.tolist()):
                confusion[y][yhat] += 1
    print("per kelas  :", "  ".join(
        f"{c} {confusion[i][i] / sum(confusion[i]):.3f}" for i, c in enumerate(classes)))

    # Row = true class, column = prediction, as shares of the row. The
    # off-diagonal cells show where each class's mistakes go, e.g. whether
    # sad takes its errors from neutral.
    print("\nconfusion (baris = asli, kolom = tebakan):")
    print(f"{'':10}" + "".join(f"{c[:7]:>9}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:10}" + "".join(f"{confusion[i][j] / sum(confusion[i]):9.3f}" for j in range(n)))

