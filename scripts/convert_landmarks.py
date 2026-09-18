#!/usr/bin/env python3
"""Convert the Landmark MLP to Core ML and prove it still agrees.

    python scripts/convert_landmarks.py     # after convert_coreml.py

Writes models/LandmarkClassifier.mlpackage. Input "points": the 152 numbers
extract_landmarks.landmarks() makes (76 Vision points, centred between the
pupils, divided by the pupil distance). The standardisation with train's
mean/std and the softmax live inside the model, so Swift cannot get them
wrong. The pupil step stays in Swift, because it needs Vision's regions.

Float32, unlike the CNN: 0.5 M parameters run in well under a millisecond
either way, so float16 would only add a gap to explain.

Checked on FER valid (PyTorch vs Core ML), then on the golden webcam crops
convert_coreml.py saved: for each, Python's 152 points, the landmark
probabilities and the fused ones go into data/webcam/golden-48/landmarks.json,
the reference the Swift side must reproduce.
"""
import json
from pathlib import Path

import coremltools as ct
import cv2
import numpy as np
import torch
import torch.nn as nn

from extract_landmarks import landmarks
from train_landmarks import CLASSES, FER, LandmarkMLP
from webcam_test import ALPHA

CKPT = "models/landmark-mlp.pt"
OUT = "models/LandmarkClassifier.mlpackage"
GOLDEN = Path("data/webcam/golden-48")


class OnDevice(nn.Module):
    """152 normalised points -> class probabilities, standardisation inside."""

    def __init__(self, mlp, mean, std):
        super().__init__()
        self.mlp = mlp
        self.register_buffer("mean", mean.view(1, -1))
        self.register_buffer("std", std.view(1, -1))

    def forward(self, x):
        return torch.softmax(self.mlp((x - self.mean) / self.std), dim=1)


if __name__ == "__main__":
    ck = torch.load(CKPT, map_location="cpu")
    mlp = LandmarkMLP(dropout=ck["dropout"])
    mlp.load_state_dict(ck["state_dict"])
    wrapper = OnDevice(mlp.eval(), ck["mean"], ck["std"]).eval()

    traced = torch.jit.trace(wrapper, torch.rand(1, 152))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="points", shape=(1, 152))],
        outputs=[ct.TensorType(name="probs")],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT32,
    )
    mlmodel.short_description = "Facial expression from 76 Vision face points (Landmark MLP, run 19)."
    mlmodel.input_description["points"] = ("x0, y0, ... x75, y75 of VNFaceLandmarks2D.allPoints, "
                                           "minus the pupils' midpoint, divided by the pupil distance")
    mlmodel.output_description["probs"] = "softmax over " + ", ".join(CLASSES)
    mlmodel.user_defined_metadata["classes"] = ",".join(CLASSES)
    mlmodel.user_defined_metadata["source_checkpoint"] = CKPT
    mlmodel.user_defined_metadata["fusion_alpha_cnn"] = str(ALPHA)
    mlmodel.save(OUT)
    print(f"wrote {OUT}  (float32)  classes {CLASSES}")

    d = np.load(FER)
    x = d["x"][d["found"] & (d["split"] == "valid")]
    with torch.no_grad():
        p_torch = wrapper(torch.from_numpy(x)).numpy()
    p_ml = np.stack([np.asarray(mlmodel.predict({"points": row[None]})["probs"]).reshape(-1) for row in x])
    print(f"FER valid ({len(x)} faces)  max |Δprob| {np.abs(p_torch - p_ml).max():.2e}  "
          f"same class {(p_torch.argmax(1) == p_ml.argmax(1)).mean():.4f}")

    expected = json.loads((GOLDEN / "expected.json").read_text())
    ref, missing = {}, 0
    for name, meta in sorted(expected.items()):
        feat = landmarks(cv2.imread(str(GOLDEN / name), cv2.IMREAD_GRAYSCALE))
        if feat is None:
            missing += 1
            continue
        p_lm = np.asarray(mlmodel.predict({"points": feat[None]})["probs"]).reshape(-1)
        p_cnn = np.array([meta["probs_coreml"][c] for c in CLASSES])
        ref[name] = {"points": feat.tolist(),
                     "probs_landmark": dict(zip(CLASSES, map(float, p_lm))),
                     "probs_fused": dict(zip(CLASSES, map(float, ALPHA * p_cnn + (1 - ALPHA) * p_lm)))}
    (GOLDEN / "landmarks.json").write_text(json.dumps(ref, indent=2))
    print(f"golden crops: points on {len(ref)}/{len(expected)} (missing {missing})  "
          f"-> {GOLDEN}/landmarks.json, the reference for Swift")
