#!/usr/bin/env python3
"""Score audio checkpoints on one split, and average them across seeds.

    python scripts/evaluate_audio.py models/audio-a2b.pt models/audio-a2b-s1.pt \
        models/audio-a2b-s2.pt --split avp-test

--split is valid or test (the bucket) or avp-valid or avp-test (AVP, 14
people each). The test splits are opened once, on Day 8.

Each checkpoint is rebuilt with the padding and normalisation it was
trained with, read from the checkpoint itself, so the pipeline cannot
drift from training. Several checkpoints are several seeds of one
configuration: the result is their mean ± sd, never the best seed,
because picking a seed on the test set would make its number optimistic
all over again.
"""
import argparse
import statistics as st
from collections import Counter
from pathlib import Path

import torch

from audio_features import CSV, BeatboxAudio
from audio_labels import CLASSES
from train_audio import AVP_CSV, AudioCNN


def score(ckpt_path, split, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    model = AudioCNN()
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    # Noise padding draws random filler; a fixed seed makes reruns identical.
    torch.manual_seed(0)
    avp = split.startswith("avp")
    ds = BeatboxAudio(split, AVP_CSV if avp else CSV,
                      pad=ck.get("pad", "zero"), norm=ck.get("norm", "none"))
    with torch.no_grad():
        pred = model(ds.x.to(device)).argmax(1).cpu().tolist()
    # AVP is scored per original label, so closed and open hihat stay apart.
    groups = [r["avp_label"] for r in ds.rows] if avp else [CLASSES[y] for y in ds.y.tolist()]
    truth = ds.y.tolist()
    hit, tot, confusion = Counter(), Counter(), Counter()
    for g, t, p in zip(groups, truth, pred):
        tot[g] += 1
        hit[g] += t == p
        confusion[(g, CLASSES[p])] += 1
    recall = {g: hit[g] / tot[g] for g in sorted(tot)}
    acc = sum(t == p for t, p in zip(truth, pred)) / len(truth)
    return acc, recall, confusion, tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--split", required=True, choices=["valid", "test", "avp-valid", "avp-test"])
    args = p.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    accs, recalls, confusion, totals = [], [], Counter(), Counter()
    print(f"split {args.split} · {len(args.checkpoints)} checkpoint")
    for path in args.checkpoints:
        acc, recall, conf, tot = score(path, args.split, device)
        accs.append(acc)
        recalls.append(recall)
        confusion.update(conf)
        totals.update(tot)
        print(f"  {Path(path).stem:22} acc {acc:.3f}  | " + "  ".join(f"{g} {r:.2f}" for g, r in recall.items()))

    groups = list(recalls[0])
    sd = lambda xs: st.stdev(xs) if len(xs) > 1 else 0.0
    print(f"\n=> acc {st.mean(accs):.3f} ± {sd(accs):.3f}")
    for g in groups:
        xs = [r[g] for r in recalls]
        print(f"   {g:8} recall {st.mean(xs):.3f} ± {sd(xs):.3f}")

    print("\nconfusion, dijumlah dari semua checkpoint (baris = asli, kolom = tebakan, bagian dari baris):")
    print(f"{'':9}" + "".join(f"{c:>9}" for c in CLASSES))
    for g in groups:
        print(f"{g:9}" + "".join(f"{confusion[(g, c)] / totals[g]:9.3f}" for c in CLASSES))


if __name__ == "__main__":
    main()
