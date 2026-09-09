# INVESTIGATE — Kanban tasks

Copy-paste ke kartu Miro. Format ikut slide Academy: one verb one outcome ·
named owner · time budget · done condition.

Owner semua: **Pafras** (solo). Window: 9 Sep 2026 (Hari 3) — dipadatkan
jadi satu hari karena pivot arah proyek. Total estimate: 8 jam.

---

## 1. Define the five ML terms in my own words, using my own project

- **Priority:** High
- **Estimate:** 1
- **Start:** Sep 9, 2026 · **End:** Sep 9, 2026

**Description**

**Question:** What is a model, a class, a label, a sample, and a feature — stated in terms of facial expressions and beats, not textbook abstractions?

**Where to look:** PyTorch 60-minute blitz, Apple's Core ML docs intro, Google ML Glossary. Skip anything longer than a page.

**Done when:** five terms written down, each in one sentence, each using a concrete example from this project (e.g. "a sample is one 48×48 grayscale photo of a face labelled `happy`").

---

## 2. Map the end-to-end steps of training a model, before writing any code

- **Priority:** High
- **Estimate:** 1
- **Start:** Sep 9, 2026 · **End:** Sep 9, 2026

**Description**

**Question:** What are the actual steps from a folder of face images to a trained model file, and what artifact does each step produce?

**Where to look:** PyTorch training-loop tutorial, torchvision `ImageFolder` docs.

**Done when:** one diagram or numbered list covering images → transforms → dataset split → training loop → checkpoint, with the file format that comes out of each step named.

---

## 3. Pick the facial-expression dataset and stop looking

- **Priority:** High
- **Estimate:** 2
- **Start:** Sep 9, 2026 · **End:** Sep 9, 2026

**Description**

**Question:** Which public dataset gets a working classifier fastest — FER2013, RAF-DB, or CK+?

**Where to look:** Kaggle (FER2013), the RAF-DB request page, papers-with-code facial-expression-recognition leaderboard. Timebox this hard — a "good enough" dataset that unblocks the whole pipeline beats a perfect one found on day 6.

**Done when:** one dataset downloaded and loading in a DataLoader, with its licence checked and its sample count per class written down. The emotion class list is fixed by the end of this task.

---

## 4. Fix which emotions map to which beat sounds

- **Priority:** High
- **Estimate:** 1
- **Start:** Sep 9, 2026 · **End:** Sep 9, 2026

**Description**

**Question:** Which emotions can a user actually perform on demand, in front of a webcam, fast enough to play a rhythm?

**Where to look:** your own face in Photo Booth. Try holding each of the seven FER classes for one second and switching between them. Disgust and fear are the usual casualties.

**Done when:** a fixed mapping table (emotion → `K` / `H` / `S`) with 3–4 entries, and the discarded classes named with the reason. Classes you can't perform reliably are dead weight in both the model and the game.

---

## 5. Decide whether to train from scratch or start from a pre-trained model

- **Priority:** Medium
- **Estimate:** 1
- **Start:** Sep 9, 2026 · **End:** Sep 9, 2026

**Description**

**Question:** Is fine-tuning a pre-trained backbone (ResNet18, MobileNet) acceptable for this challenge, or does the Academy expect a model trained from scratch?

**Where to look:** the challenge brief, and one message to the mentor. Then the model cards of the pre-trained options.

**Done when:** a written decision with its reason, plus the mentor's answer on whether transfer learning counts as "training a model". Either way the plan compares both, so this changes the framing rather than the work.

---

## 6. Confirm the conversion path from PyTorch to a file macOS can run

- **Priority:** High
- **Estimate:** 1
- **Start:** Sep 9, 2026 · **End:** Sep 9, 2026

**Description**

**Question:** How does a trained PyTorch model become something Core ML runs, and where does ONNX fit — is it needed at all?

**Where to look:** coremltools documentation, "Converting from PyTorch" page.

**Done when:** the conversion path is written in one line, and one throwaway toy model has actually been converted and loaded in Xcode — proving the toolchain works before there is a real model to lose.

---

## 7. (Optional) Work out how the emotion model and Foundation Models hand off to each other

- **Priority:** Low — Foundation Models is an optional extra, not a scheduled deliverable
- **Estimate:** 1
- **Start:** Sep 9, 2026 · **End:** Sep 9, 2026

**Description**

**Question:** How do two models run in one app — what exactly does the emotion classifier's output pass to Foundation Models, and in what shape?

**Where to look:** Apple's Foundation Models framework docs, WWDC session on on-device model APIs.

**Done when:** the handoff is written as one line of pseudo-Swift showing the data type crossing the boundary (e.g. `([Emotion], Score) -> String`), and the limits of Foundation Models are noted in two bullets.
