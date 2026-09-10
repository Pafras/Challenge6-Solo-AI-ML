# Project Context — Emotion-to-Beatbox (Academy Solo Challenge)

Read this before making architectural suggestions or asking "what is this
project for". The reasoning behind each decision is included so settled
trade-offs don't get re-litigated.

## The actual goal

**The point of this project is to learn how to train a model properly — with
a lot of options genuinely explored and compared.** Architecture,
hyperparameters, augmentation, class balance, transfer learning: each one
tried, measured, logged, compared. Not "get one model that works".

The macOS app exists because building one is part of the challenge, and
because a demo you can play beats a notebook full of numbers. It is
deliberately kept simple. **If the app starts eating training days, cut the
app, not the training.**

**Final submission is Thursday 17 September**, a day earlier than the sprint's nominal end. Days 3–8 are training (6 days, untouched). Days 9–10 are the app (2 days). Day 11 is submission day, not a working day — nothing may be stacked on it.

## What the product is

A macOS app where facial expressions control a beatbox, plus a quest system
where the user recreates a target beat by performing an expression sequence.

```text
FACE → EMOTION → BEAT → AUDIO → QUEST → SCORE
```

## Context: what kind of project this is

Solo challenge at an Apple Developer Academy (Bali), 7–17 September 2026.
Final submission Thursday 17 September. Academy framework: **Engage (1d) → Investigate (2d) → Act (2d,
end of week 1) → Act (5d, week 2) → Cooldown**. The Sat–Sun between week 1
and week 2 is informal buffer.

Academy ingredients, as weighted by the person doing the work (not by the
brief's star rating):

- **Training models with PyTorch** — the core. Everything else is secondary.
- **Core ML** — on the critical path, because a PyTorch model cannot run in
  a macOS app without conversion. One day, convert and verify.
- **Foundation models & native AI** — optional. Noted as a possible addition
  in `CHALLENGE.md` under "Opsional", not scheduled.

## How the direction got here (read before suggesting alternatives)

1. Started as **hand-gesture recognition** with Create ML.
2. **Mentor feedback #1:** Create ML's built-in classifier is too close to
   drag-and-drop — no architecture design, no training loop, no
   hyperparameter work. Too small for a solo challenge.
3. **Mentor feedback #2:** train in **PyTorch**, convert with `coremltools`.
   Firm decision. PyTorch over TensorFlow because Apple's `coremltools` has a
   first-class, actively maintained PyTorch path, and PyTorch transfers
   better to future ML work.
4. **Mentor feedback #3:** gesture recognition isn't novel — iPhone already
   does it natively. Pivot to a personal hobby instead.
5. Went to **guitar chord recognition**. Dropped: weak dataset situation,
   and the output was a static chart rather than something interactive.
6. **Current: Emotion-to-Beatbox.** Facial expression as input (large public
   datasets exist), music as output, quest layer to make it demoable. Full
   spec in `docs/spec-emotion-beatbox.md`.

## Technical approach

- **Input:** webcam frame → face detection → face crop.
- **Two models, on the mentor's advice (10 Sep).** First: a
  facial-expression classifier, image to emotion, which ships in the app.
  Second: a beatbox audio classifier, sound to kick/hihats/snare/clap,
  trained as a second exercise from the 576 local wav files whose filename
  prefix is the label. The audio model deliberately does **not** ship — the
  app has no mic input, and adding one plus a second Core ML conversion
  costs a day that does not exist. Its deliverable is a model card and a row
  in the experiment table.
- **Architecture is not pre-decided** for either — comparing small CNN vs
  ResNet18 vs MobileNet, from-scratch vs fine-tuned, *is* the exercise.
- **Training loop:** hand-written, not `Trainer`-wrapped. Config-driven so
  swapping an option doesn't mean editing code.
- **Conversion:** `coremltools` PyTorch path → `.mlpackage`, verified
  numerically against PyTorch on the same input.
- **macOS app:** SwiftUI + Vision (face detection) + Core ML (inference) +
  AVFoundation (audio). Kept small.
- **Beat:** symbolic tokens (`K` kick, `H` hi-hat, `S` snare). Rule-based
  emotion → beat mapping, a plain lookup table. No ML beat generator — it
  would add nothing to the learning goal.
- **Audio engine:** sample playback on a BPM clock. Not AI, doesn't need
  to be.
- **Quest:** 2–3 hardcoded target sequences, compared against the detected
  sequence, scored on sequence match.

## Experiment discipline

This is where the learning actually happens, so it has rules:

- One run = one row in `docs/experiments.md`. Unlogged runs don't count.
- Change one variable per run. Change two and the result is uninterpretable.
- Seed and split are fixed and saved to disk. Runs are only comparable if
  the data underneath them is identical.
- The test set is opened once, on Day 8. All tuning uses the val set.
- No MLflow, no Weights & Biases. A markdown table is enough for ~30 runs.

## Scope discipline

**P0:** the training experiments and the log; expression recognition working
in real time; a simple app where an expression changes the sound; the audio
classifier trained and measured.

**P1:** quest + scoring — now a stretch goal on Day 10, not a scheduled
deliverable. Free mode (expression changes the sound live) carries the demo
on its own.

**Cut before cutting training days:** quest complexity, UI polish, timing
accuracy, Foundation Models, emotion intensity, ML beat generator.

Explicitly do NOT build: a custom face detector, a cloud backend, user
accounts, a database, a progression system, 3D graphics.

## Known risks

- **Fiddling without logging.** The number-one risk for this specific goal.
  Twenty runs and no results table means nothing was learned, just time
  spent.
- **Prediction jitter.** Real-time predictions flicker between frames even
  when the user holds still. Temporal smoothing (majority vote over a short
  window + confidence threshold) is mandatory. Number-one cause of the demo
  feeling broken.
- **Preprocessing parity.** Face crop, resize and normalisation in Swift
  must match training in PyTorch exactly. A mismatch drops accuracy
  silently, with no error. Note that Core ML ML Program runs float16 by
  default, so some gap against PyTorch is expected and is not the bug you
  are looking for. How much depends on depth: around 1e-4 for the toy CNN
  (`scripts/test_conversion.py`), around 1e-2 on the logits for MobileNetV3
  (`scripts/check_coreml.py`), because a deep network compounds rounding
  across many layers. Float32 conversion of MobileNetV3 matches to ~1e-7, so
  the operations translate faithfully. A trained model's confident
  predictions have margins far wider than 1e-2; only genuinely ambiguous
  faces could flip, and temporal smoothing absorbs those. Shipping float32
  instead is a one-flag change if it ever matters.
- **The app eating the training.** Days 9–11 are hard-capped. Cut app scope,
  never training days.
- **The second model eating the first.** Face experiments dropped from three
  days to two to make room. If the audio model overruns on Day 7, cut its
  experiments rather than the face model's — one audio model that merely
  works beats two half-finished ones.
- **Two days lost to the pivot, one to the deadline.** Days 1–2 went to the
  previous direction, and submission moved up to the 17th. The spec's 12 days
  are compressed into 9 (Days 3–11). The app absorbed the cut, not training.
- **No buffer on the last day.** Day 11 is submission. The model card is
  drafted on Day 8 while the numbers are fresh, and a backup demo is recorded
  on Day 10, so a slip on Wednesday doesn't mean submitting nothing.

## What's still open

- Dataset choice (FER2013 vs RAF-DB vs CK+). Decide fast, don't hunt for the
  perfect one. Day 3.
- Final emotion class list. Starts at 4 (happy / neutral / angry / surprise),
  chosen for what a user can actually perform on demand in front of a webcam,
  not for what scores best.
- Whether the mentor counts transfer learning as "training a model" — worth
  asking, but the plan compares scratch and pretrained either way, so the
  answer changes framing rather than work.
