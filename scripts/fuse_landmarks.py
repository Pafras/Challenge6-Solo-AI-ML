#!/usr/bin/env python3
"""Step 3 of the landmark run: late fusion of the CNN and the Landmark MLP.

    python scripts/fuse_landmarks.py

Each model scores the face on its own, then their probabilities are mixed:

    p = a * p_cnn + (1 - a) * p_landmark

a = 1 is the shipped CNN alone, a = 0 the Landmark MLP alone. a is swept
0..1 in steps of 0.05 on FER valid. Faces Vision found no points on keep
the CNN's answer whatever a is, which is also what the app would do.

Criterion, set on 15 Sep before any fused number was seen:
    FER valid up >= 1 point over the CNN alone, AND
    on erin, angry or sad recall up without erin's total going down.
a is chosen on FER valid and reported there, so that gain is a little
optimistic (the best of 21 tries). Erin is never used to choose a.
"""
import numpy as np
import torch
from PIL import Image

from show_batch import build_transform
from train import build_model
from train_landmarks import CLASSES, FER, OWN, LandmarkMLP

CNN = "models/mobilenet-finetune-bersih.pt"
MLP = "models/landmark-mlp.pt"
ALPHAS = np.round(np.linspace(0, 1, 21), 2)


@torch.no_grad()
def cnn_probs(ck, paths, device):
    net = build_model(ck["arch"], n_classes=len(CLASSES))
    net.load_state_dict(ck["state_dict"])
    net.eval().to(device)
    tf = build_transform(ck["image_size"], ck["channels"])
    out = []
    for i in range(0, len(paths), 256):
        x = torch.stack([tf(Image.open(p)) for p in paths[i:i + 256]]).to(device)
        out.append(torch.softmax(net(x), 1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def mlp_probs(x):
    ck = torch.load(MLP, map_location="cpu")
    net = LandmarkMLP(dropout=ck["dropout"])
    net.load_state_dict(ck["state_dict"])
    net.eval()   # forget this and dropout + BatchNorm give a different answer every call
    return torch.softmax(net((torch.from_numpy(x) - ck["mean"]) / ck["std"]), 1).numpy()


def rows(npz, split):
    d = np.load(npz)
    keep = d["split"] == split
    return list(d["path"][keep]), d["label"][keep], d["found"][keep], d["x"][keep]


def recall(pred, y):
    return {c: (pred[y == i] == i).mean() for i, c in enumerate(CLASSES)}


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(CNN, map_location="cpu")
    assert ck["classes"] == CLASSES, "class order differs between the two models"

    sets = {}
    for name, npz, expected in [("FER valid", FER, ck["val_acc"]), ("erin", OWN, ck["own_val_acc"])]:
        paths, y, found, x = rows(npz, "valid")
        pc = cnn_probs(ck, paths, device)
        pl = np.where(found[:, None], mlp_probs(x), pc)
        cnn_ok, lm_ok = pc.argmax(1) == y, pl.argmax(1) == y
        # The CNN must score what it scored when it was trained. If not, the
        # images here are preprocessed differently and every number below is off.
        assert abs(cnn_ok.mean() - expected) < 0.005, f"CNN {cnn_ok.mean():.3f}, checkpoint says {expected:.3f}"
        print(f"{name}: {len(y)} wajah · CNN {cnn_ok.mean():.3f} · landmark {lm_ok[found].mean():.3f} "
              f"(di {found.sum()} yang ada titiknya)")
        # Fusion can only help where the two disagree. If the MLP is right
        # almost only where the CNN already is, there is nothing to gain.
        print(f"  CNN salah, landmark benar: {(~cnn_ok & lm_ok).mean():.1%} · "
              f"CNN benar, landmark salah: {(cnn_ok & ~lm_ok).mean():.1%}")
        sets[name] = pc, pl, y

    mix = lambda a, pc, pl: a * pc + (1 - a) * pl
    pc, pl, y = sets["FER valid"]
    accs = [(mix(a, pc, pl).argmax(1) == y).mean() for a in ALPHAS]
    print("\nsapuan α di FER valid (1.00 = CNN saja):")
    for i in range(0, len(ALPHAS), 7):
        print("  " + "   ".join(f"α {a:.2f} {acc:.3f}" for a, acc in zip(ALPHAS[i:i + 7], accs[i:i + 7])))
    best_a = max(zip(accs, ALPHAS))[1]   # a tie goes to the larger a: less landmark, less change

    print()
    result = {}
    for name, (pc, pl, y) in sets.items():
        for label, a in [("CNN saja", 1.0), (f"gabungan α {best_a:.2f}", best_a)]:
            pred = mix(a, pc, pl).argmax(1)
            result[name, a] = (pred == y).mean(), recall(pred, y)
            print(f"{name:9} {label:17} {result[name, a][0]:.3f}   "
                  + " · ".join(f"{c} {r:.2f}" for c, r in result[name, a][1].items()))

    fer_gain = result["FER valid", best_a][0] - result["FER valid", 1.0][0]
    (e_cnn, r_cnn), (e_mix, r_mix) = result["erin", 1.0], result["erin", best_a]
    checks = {
        f"FER valid naik >= 1 poin ({fer_gain * 100:+.1f})": fer_gain >= 0.01,
        f"erin angry atau sad naik (angry {r_cnn['angry']:.2f} -> {r_mix['angry']:.2f}, "
        f"sad {r_cnn['sad']:.2f} -> {r_mix['sad']:.2f})": r_mix["angry"] > r_cnn["angry"] or r_mix["sad"] > r_cnn["sad"],
        f"erin total gak turun ({e_cnn:.3f} -> {e_mix:.3f})": e_mix >= e_cnn,
    }
    print("\nkriteria:")
    for text, ok in checks.items():
        print(f"  {'lolos' if ok else 'GAGAL'}  {text}")
    print("→ " + ("LOLOS: gabungan layak masuk app" if all(checks.values())
                  else "GAK LOLOS: app tetap CNN saja, landmark cuma visualisasi"))
