#!/usr/bin/env python3
"""Prove the PyTorch -> Core ML toolchain works, before there is a real model.

    python scripts/test_conversion.py

Trains nothing. Builds a throwaway CNN shaped like the real one, converts it,
runs both, and compares the numbers. If this passes, Day 9 is a known path
instead of a gamble. If it fails, it fails now, while the only thing lost is
a random-weight model.

Conversion path (no ONNX in it):
    nn.Module -> torch.jit.trace -> ct.convert(...) -> .mlpackage

Expect a gap around 1e-4, not 1e-6: Core ML ML Program runs float16 by
default. Pass compute_precision=ct.precision.FLOAT32 if that ever matters.
"""
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

# Same shape as the planned classifier: 48x48 grayscale face crop, 4 emotions.
SHAPE = (1, 1, 48, 48)
TOLERANCE = 1e-3


class TinyCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 12 * 12, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    model = TinyCNN().eval()
    x = torch.randn(*SHAPE)

    with torch.no_grad():
        expected = model(x).numpy()

    traced = torch.jit.trace(model, x)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=SHAPE)],
        # Name the output. Left unnamed, coremltools invents one from the
        # graph ("var_41"), which changes whenever the architecture changes —
        # and Swift refers to outputs by name.
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.macOS13,
    )

    # predict() runs through the real Core ML runtime, not a simulation.
    got = np.array(mlmodel.predict({"input": x.numpy()})["logits"])

    diff = np.abs(expected - got).max()
    print(f"\npytorch  {np.round(expected.ravel(), 4)}")
    print(f"coreml   {np.round(got.ravel(), 4)}")
    print(f"max diff {diff:.2e}  (tolerance {TOLERANCE:.0e})")

    assert diff < TOLERANCE, f"outputs disagree by {diff:.2e}"
    print("\nok — toolchain works, no ONNX needed")


if __name__ == "__main__":
    main()
