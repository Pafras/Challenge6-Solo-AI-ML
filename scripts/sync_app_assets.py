#!/usr/bin/env python3
"""Copy what the app bundles into its Xcode folder.

    python scripts/sync_app_assets.py

The Core ML model, the pattern table and the beat samples are made by
scripts (convert_coreml.py, make_samples.py) into models/ and audio/.
The app bundles whatever sits in its synchronized folder, so this copies
them to EmotionBeatbox/EmotionBeatbox/Resources/. Run it after either
script; it only copies, it never generates.
"""
import shutil
from pathlib import Path

DEST = Path("EmotionBeatbox/EmotionBeatbox/Resources")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ("EmotionClassifier.mlpackage", "LandmarkClassifier.mlpackage"):
        model = DEST / name
        if model.exists():
            shutil.rmtree(model)
        shutil.copytree(f"models/{name}", model)
    shutil.copy2("audio/patterns.json", DEST / "patterns.json")
    for old in DEST.glob("*.wav"):        # a take that no longer exists must not linger
        old.unlink()
    wavs = sorted(Path("audio/samples").glob("*.wav"))
    for wav in wavs:
        shutil.copy2(wav, DEST / wav.name)
    print(f"{DEST}: EmotionClassifier + LandmarkClassifier .mlpackage, patterns.json, {len(wavs)} samples")


if __name__ == "__main__":
    main()
