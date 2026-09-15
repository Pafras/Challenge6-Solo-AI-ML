#!/usr/bin/env python3
"""Convert the final face model to Core ML and prove it still agrees.

    python scripts/convert_coreml.py              # float16, what ships
    python scripts/convert_coreml.py --fp32

Writes models/EmotionClassifier.mlpackage. The Swift side only has to hand
it a 48x48 grayscale face crop: the rest of the training preprocessing
(up to 224, three channels, ImageNet normalisation) and the softmax live
inside the model. Every step that stays in Swift is a step that can
silently differ from training, so as few as possible stay there.

Checked three ways, each against the one before, so a gap can be traced:
    A  the training pipeline: PIL resize to 224, torchvision transform
    B  the wrapper in PyTorch: the same, with the resize done by
       F.interpolate, the way Core ML will do it
    C  the .mlpackage, run through the real Core ML runtime
on the whole FER2013 valid split and on the golden webcam frames.
"""
import argparse
import json
from pathlib import Path

import coremltools as ct
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from show_batch import FER2013, build_transform
from train import build_model
from webcam_test import crop_box

CKPT = "models/mobilenet-finetune-bersih.pt"
OUT = "models/EmotionClassifier.mlpackage"
GOLDEN = Path("data/webcam")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class OnDevice(nn.Module):
    """48x48 grey in [0, 1] -> class probabilities, everything else inside."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        return torch.softmax(self.net(x), dim=1)


def face48(frame_bgr, box):
    """The live pipeline up to the model input: crop, grey, down to 48."""
    x0, y0, x1, y1 = box
    grey = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (48, 48), interpolation=cv2.INTER_AREA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp32", action="store_true", help="ship float32 instead of float16")
    args = p.parse_args()

    ck = torch.load(CKPT, map_location="cpu")
    classes = ck["classes"]
    net = build_model(ck["arch"], n_classes=len(classes))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    wrapper = OnDevice(net).eval()

    traced = torch.jit.trace(wrapper, torch.rand(1, 1, 48, 48))
    precision = ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16
    mlmodel = ct.convert(
        traced,
        # An image input: Swift passes a one-channel 48x48 CVPixelBuffer and
        # Core ML scales 0-255 to 0-1, exactly as ToTensor did in training.
        inputs=[ct.ImageType(name="face48", shape=(1, 1, 48, 48),
                             color_layout=ct.colorlayout.GRAYSCALE, scale=1 / 255.0)],
        # Named, because Swift refers to outputs by name.
        outputs=[ct.TensorType(name="probs")],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=precision,
    )
    mlmodel.short_description = ("Facial expression from a 48x48 grayscale face crop "
                                 "(Vision box widened by the 'fer' margin).")
    mlmodel.input_description["face48"] = "48x48 grayscale crop, Vision face box + fer margin, area-resized"
    mlmodel.output_description["probs"] = "softmax over " + ", ".join(classes)
    mlmodel.user_defined_metadata["classes"] = ",".join(classes)
    mlmodel.user_defined_metadata["source_checkpoint"] = CKPT
    mlmodel.user_defined_metadata["threshold"] = "0.6"
    mlmodel.save(OUT)
    print(f"wrote {OUT}  ({'float32' if args.fp32 else 'float16'})  classes {classes}")

    # --- FER2013 valid: A vs B vs C -------------------------------------
    ds = FER2013("valid", build_transform(ck["image_size"], ck["channels"]),
                 csv_path=f"data/splits/fer2013_{len(classes)}class.csv")
    A, B, C, y = [], [], [], []
    with torch.no_grad():
        for i, row in enumerate(ds.rows):
            img = Image.open(row["path"])
            x224, label = ds[i]
            A.append(torch.softmax(net(x224.unsqueeze(0)), 1)[0].numpy())
            x48 = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).view(1, 1, 48, 48)
            B.append(wrapper(x48)[0].numpy())
            C.append(np.asarray(mlmodel.predict({"face48": img})["probs"]).reshape(-1))
            y.append(label)
    A, B, C, y = map(np.array, (A, B, C, y))

    def compare(name, P, Q):
        agree = (P.argmax(1) == Q.argmax(1)).mean()
        print(f"  {name:24} max |Δprob| {np.abs(P - Q).max():.4f}  mean {np.abs(P - Q).mean():.5f}  "
              f"same class {agree:.4f}")

    print(f"\nFER2013 valid ({len(y)} images)")
    print(f"  accuracy  A {(A.argmax(1) == y).mean():.4f}  B {(B.argmax(1) == y).mean():.4f}  "
          f"C {(C.argmax(1) == y).mean():.4f}")
    compare("A vs B (resize method)", A, B)
    compare("B vs C (Core ML)", B, C)
    compare("A vs C (total)", A, C)

    # --- Golden webcam frames: the live path, and a reference for Swift --
    golden = sorted(GOLDEN.glob("golden-*.png"))
    if not golden:
        return
    out_dir = GOLDEN / "golden-48"
    out_dir.mkdir(exist_ok=True)
    ref, gA, gC = {}, [], []
    tf = build_transform(ck["image_size"], ck["channels"])
    with torch.no_grad():
        for f in golden:
            meta = json.loads(f.with_suffix(".json").read_text())
            small = face48(cv2.imread(str(f)), meta["box"])
            cv2.imwrite(str(out_dir / f.name), small)
            pa = torch.softmax(net(tf(Image.fromarray(small)).unsqueeze(0)), 1)[0].numpy()
            pc = np.asarray(mlmodel.predict({"face48": Image.fromarray(small)})["probs"]).reshape(-1)
            gA.append(pa)
            gC.append(pc)
            ref[f.name] = {"box": meta["box"], "margin": meta["margin"],
                           "probs_coreml": dict(zip(classes, map(float, pc)))}
    gA, gC = np.array(gA), np.array(gC)
    print(f"\ngolden webcam frames ({len(golden)})")
    compare("A vs C (live path)", gA, gC)
    # What the Swift app must reproduce: its own 48x48 crop of each frame
    # and Core ML's probabilities on it (±0.02 is float16 noise).
    (out_dir / "expected.json").write_text(json.dumps(ref, indent=2))
    print(f"  reference for Swift: {out_dir}/expected.json + the 48x48 crops")


if __name__ == "__main__":
    main()
