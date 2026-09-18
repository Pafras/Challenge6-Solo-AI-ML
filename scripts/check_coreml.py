#!/usr/bin/env python3
"""Check that a real architecture from train.py survives Core ML conversion.

    python scripts/check_coreml.py mobilenet

test_conversion.py proved the toolchain on a plain two-layer CNN. The
pretrained models contain layers that one never exercised (MobileNetV3 uses
hardswish and squeeze-excitation blocks), so each candidate gets checked on
its own before it is chosen, not on Day 9.

Weights are the pretrained ones with an untrained 4-class head. That is
enough: conversion is about whether the operations translate, not about
what the weights have learned.

Correctness is judged in float32, where a faithful conversion matches
PyTorch almost exactly. Float16 — Core ML's default, and what ships — is
reported but not asserted on: an untrained head produces near-tied logits,
and float16 drift can reorder a near-tie without anything being wrong.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

from train import ARCH, build_model

N_INPUTS = 8


def main(arch):
    spec = ARCH[arch]
    shape = (1, spec["channels"], spec["size"], spec["size"])

    torch.manual_seed(0)
    model = build_model(arch).eval()
    example = torch.rand(*shape)

    traced = torch.jit.trace(model, example)

    def convert(precision):
        return ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=shape)],
            outputs=[ct.TensorType(name="logits")],
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=precision,
        )

    m32 = convert(ct.precision.FLOAT32)
    m16 = convert(ct.precision.FLOAT16)

    d32, d16, agree16 = [], [], 0
    for _ in range(N_INPUTS):
        x = torch.rand(*shape)
        with torch.no_grad():
            expected = model(x).numpy()
        r32 = np.array(m32.predict({"input": x.numpy()})["logits"])
        r16 = np.array(m16.predict({"input": x.numpy()})["logits"])
        d32.append(np.abs(expected - r32).max())
        d16.append(np.abs(expected - r16).max())
        agree16 += int(expected.argmax() == r16.argmax())

    def size_mb(mlmodel):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.mlpackage"
            mlmodel.save(str(path))
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6

    print(f"\narch              {arch}")
    print(f"input             {shape}")
    print(f"float32 max diff  {max(d32):.2e}   {size_mb(m32):.1f} MB")
    print(f"float16 max diff  {max(d16):.2e}   {size_mb(m16):.1f} MB   "
          f"(same class {agree16}/{N_INPUTS} — untrained head, near-ties expected)")

    assert max(d32) < 1e-4, f"float32 conversion drifts by {max(d32):.2e}: an op did not translate"
    print("\nok — every operation converts faithfully")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mobilenet")
