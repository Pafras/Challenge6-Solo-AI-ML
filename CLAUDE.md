# Project Context — Guitar Chord Recognizer (Academy Solo Challenge)

This file exists so Claude Code has full context on this project without needing
it re-explained. It summarizes a planning discussion that happened before any
code was written. Read this before making architectural suggestions or asking
"what is this project for" — the reasoning behind each decision is included so
you don't re-litigate settled trade-offs.

## What this project is

An iOS app that listens to guitar playing, recognizes the chords being played
in real time using a custom-trained ML model, and automatically turns a
recorded jam session into a chord chart — instead of the player having to
remember or manually transcribe what they played.

**Challenge statement:**
> Learn how to train a model that listens to my guitar playing and
> automatically turns it into a chord chart.

## Context: what kind of project this is

This is a solo challenge at an Apple Developer Academy (Bali), running
7–18 September 2026 (12 days). The Academy's own framework structures the
work as: **Engage (1 day) → Investigate (2 days) → Act (2 days, end of week 1)
→ Act (5 days, week 2) → Cooldown**. The two extra weekend days (Sat–Sun
between week 1 and week 2) are being used as informal buffer, not part of the
official structure.

The Academy also provided three "ingredients" this challenge must combine,
each rated by how central it should be:
- **Foundation models & native AI capabilities** (3★) — what Apple already
  offers out of the box, explored before training anything custom.
- **Training models with Create ML** (4★, originally) — building a model from
  your own data.
- **Integrating models with Core ML** (3★) — getting a trained model running
  inside the app.

## How the direction got here (read this before suggesting alternatives)

The project went through several pivots during planning. Each one is recorded
here so the reasoning isn't lost:

1. **Started as hand-gesture recognition** (static hand poses via Vision's
   hand-pose landmarks, classified with Create ML) for a "silent
   communication" use case. This was the original plan built around the
   three ingredients above.
2. **Mentor feedback #1:** the Create ML approach was too small in scope for
   a solo challenge — Create ML's built-in classifier is close to
   drag-and-drop, with little real ML engineering (no architecture design,
   no training loop, no hyperparameter work).
3. **Mentor feedback #2:** train the model in **PyTorch or TensorFlow**
   instead of Create ML's no-code trainer, then convert to Core ML with
   `coremltools`. This became a firm decision — **PyTorch was chosen** over
   TensorFlow because Apple's `coremltools` has a more actively maintained,
   first-class PyTorch conversion workflow, and PyTorch is more transferable
   to future AI/ML work.
4. **Mentor feedback #3:** hand-gesture recognition is arguably not novel
   enough, since gesture-based interaction already exists natively on
   iPhone. Suggested pivoting to a topic drawn from a personal hobby instead.
5. **Pivoted to music** (the person's hobbies include guitar and drums).
   Guitar was chosen over drums as the primary instrument (drums remain a
   possible future extension: drum-hit classification would need an onset
   detection step before classification, which is a meaningfully different,
   two-stage pipeline).
6. Considered three product directions built on top of "recognize the
   chord being played": (a) a real-time practice companion that checks your
   playing against a target progression, (b) auto-generating a chord chart
   from a recorded jam/improvisation, (c) an ear-training quiz game.
   **(b) was chosen** — it's the most personally motivated (solves a real
   "I forgot what I just played" problem for an improvising musician), the
   easiest to demo, and doesn't require pre-authoring a "target song" before
   the feature is useful.

**Net effect on the ingredients:** Create ML is effectively replaced by
PyTorch for the actual model training (per mentor feedback #2) — Core ML is
still the integration layer, and Foundation Models is still used, now as a
natural-language layer that comments on the detected chord progression
(e.g., "this progression is a classic pop/folk pattern") rather than as the
core recognition mechanism.

## Technical approach (current plan)

- **Input:** short audio clips of guitar chords / a recorded jam session
  (via `AVAudioEngine` on-device).
- **Features:** mel-spectrogram (or similar) extracted from the audio.
- **Model:** a small CNN (or similarly lightweight architecture) trained in
  PyTorch to classify chords from the spectrogram. Kept intentionally small —
  the input is a spectrogram of a few seconds of audio, not raw video, so
  the model does not need to be large to be a legitimate ML engineering
  exercise.
- **Conversion:** `coremltools`' PyTorch conversion workflow, producing a
  `.mlmodel` / `.mlpackage` for on-device inference.
- **iOS integration:** SwiftUI app (the person's existing stack is SwiftUI
  with MVVM/MVC), Core ML for inference, a UI that renders the detected
  chord sequence as a chart.
- **Foundation Models layer:** after a chord sequence is detected, pass it
  to Apple's on-device Foundation Models framework to generate a short
  natural-language comment about the progression (style, similar songs,
  possible next chord) — free, on-device, no API key required.

## What's still open (do not assume these are decided)

- The exact 12-day / Engage–Investigate–Act–Cooldown breakdown has **not**
  been finalized for this audio-based direction — an earlier day-by-day plan
  and tracker existed for the hand-gesture version and is now obsolete.
  If asked to help plan the days, build a fresh breakdown around: audio data
  collection → spectrogram pipeline → PyTorch model training/iteration →
  coremltools conversion → SwiftUI integration → Foundation Models layer →
  device testing → demo/documentation (cooldown).
  Do **not** assume static hand-pose landmarks, Create ML, or a gesture
  vocabulary are still part of the project — they were explicitly replaced.
- Exact chord vocabulary (how many/which chords to recognize first) is not
  yet fixed. Start narrow (a handful of common open chords) and expand only
  if time allows, consistent with the "keep scope tight" lesson learned
  from the original gesture-recognition scoping mistake.
- Dataset size/collection method (self-recorded vs. any public guitar-chord
  audio datasets) has not been decided yet.
