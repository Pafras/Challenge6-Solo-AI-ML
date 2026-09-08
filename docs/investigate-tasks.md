# INVESTIGATE — Kanban tasks

Copy-paste ke kartu Miro. Format ikut slide Academy: one verb one outcome ·
named owner · time budget · done condition.

Owner semua: **Pafras** (solo). Window: 8–9 Sep 2026 (Hari 2–3).
Total estimate: 11 jam.

---

## 1. Define the five ML terms in my own words, using my own project

- **Priority:** High
- **Estimate:** 2
- **Start:** Sep 8, 2026 · **End:** Sep 8, 2026

**Description**

**Question:** What is a model, a class, a label, a sample, and a feature — stated in terms of guitar chords, not textbook abstractions?

**Where to look:** PyTorch 60-minute blitz, Apple's Core ML docs intro, Google ML Glossary. Skip anything longer than a page.

**Done when:** five terms written down, each in one sentence, each using a concrete example from this project (e.g. "a sample is one 2-second .wav of me strumming C").

---

## 2. Map the end-to-end steps of training a model, before writing any code

- **Priority:** High
- **Estimate:** 2
- **Start:** Sep 8, 2026 · **End:** Sep 8, 2026

**Description**

**Question:** What are the actual steps from raw audio to a trained model file, and what artifact does each step produce?

**Where to look:** PyTorch training-loop tutorial, torchaudio spectrogram tutorial.

**Done when:** one diagram or numbered list covering audio → mel-spectrogram → dataset split → training loop → checkpoint, with the file format that comes out of each step named.

---

## 3. Decide where the chord audio data comes from

- **Priority:** High
- **Estimate:** 3
- **Start:** Sep 8, 2026 · **End:** Sep 9, 2026

**Description**

**Question:** Is there a usable public guitar-chord dataset, or do I record it myself?

**Where to look:** Zenodo, Kaggle, Freesound, GuitarSet, papers-with-code audio-classification datasets.

**Done when:** one dataset chosen with its licence checked and its sample-count per chord written down — or a decision to self-record, with the recording setup (mic, room, clip length) specified. Either way, the chord list is fixed by the end of this task.

---

## 4. Decide whether to train from scratch or start from a pre-trained model

- **Priority:** Medium
- **Estimate:** 2
- **Start:** Sep 9, 2026 · **End:** Sep 9, 2026

**Description**

**Question:** Is using a pre-trained audio model (YAMNet, VGGish, PANNs) acceptable for this challenge, or does the Academy expect a model trained from scratch?

**Where to look:** the challenge brief, and one message to the mentor. Then the model cards of the pre-trained options.

**Done when:** a written decision with its reason, plus the mentor's answer on whether transfer learning counts as "training a model".

---

## 5. Confirm the conversion path from PyTorch to a file the iPhone can run

- **Priority:** High
- **Estimate:** 1
- **Start:** Sep 9, 2026 · **End:** Sep 9, 2026

**Description**

**Question:** How does a trained PyTorch model become something Core ML runs, and where does ONNX fit — is it needed at all?

**Where to look:** coremltools documentation, "Converting from PyTorch" page.

**Done when:** the conversion path is written in one line, and one throwaway toy model has actually been converted and loaded — proving the toolchain works before there is a real model to lose.

---

## 6. Work out how the chord model and Foundation Models hand off to each other

- **Priority:** Medium
- **Estimate:** 1
- **Start:** Sep 9, 2026 · **End:** Sep 9, 2026

**Description**

**Question:** How do two models run in one app — what exactly does the chord classifier pass to Foundation Models, and in what shape?

**Where to look:** Apple's Foundation Models framework docs, WWDC session on on-device model APIs.

**Done when:** the handoff is written as one line of pseudo-Swift showing the data type crossing the boundary (e.g. `[String] -> String`), and the limits of Foundation Models are noted in two bullets.
