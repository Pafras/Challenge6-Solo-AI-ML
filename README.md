# Emotion-to-Beatbox

A macOS app that turns **facial expressions into a beat**. The camera reads your face, a model picks the expression, and each expression plays its own beatbox pattern, looping in time.

```text
CAMERA → FACE CROP → EXPRESSION → BEAT PATTERN → AUDIO
```

Solo challenge at the Apple Developer Academy (Bali), 7–17 September 2026. The main goal was **learning to train a model properly**: architecture, hyperparameters, augmentation, class balance and transfer learning, each tried one at a time, measured and compared. The app is deliberately kept simple.

## What's in the repo

| Folder | Contents |
|---|---|
| `scripts/` | All the Python: data, training, evaluation, Core ML conversion, webcam test |
| `EmotionBeatbox/` | The macOS Xcode project (SwiftUI + Vision + Core ML + AVFoundation) |
| `audio/` | The beat pattern table (`patterns.json`) and 12 beatbox samples |
| `requirements.txt` | Python dependencies, with the reason behind each pinned version |

Datasets, checkpoints (`*.pt`) and converted models in `models/` are kept out of git because of their size. Only the copies the app bundles, in `EmotionBeatbox/EmotionBeatbox/Resources/`, are committed.

## Two models

### 1. Facial expression classifier (ships in the app)

| | |
|---|---|
| Task | face crop → **angry / happy / neutral / surprise / sad** |
| Architecture | MobileNetV3-Small, ImageNet weights, ~1.52 million parameters |
| Data | FER2013 (5 classes) + 1,200 webcam crops of 4 people, recorded with consent |
| FER2013 test | **accuracy 0.752 · macro F1 0.745** (test set opened once; random guessing is 0.20) |
| A person never trained on | 0.660 → **0.767** after fine-tuning |

Per class on the FER2013 test set:

| class | precision | recall | F1 |
|---|---|---|---|
| angry | 0.701 | 0.648 | 0.674 |
| happy | 0.901 | 0.864 | 0.882 |
| neutral | 0.685 | 0.659 | 0.672 |
| surprise | 0.831 | 0.897 | 0.863 |
| sad | 0.607 | 0.670 | 0.637 |

**What the app actually runs: late fusion.** The CNN above is combined with a Landmark MLP, which sees only 76 face points from Vision and no pixels at all:

```text
p = 0.55 · p_cnn + 0.45 · p_landmark
```

The two models get different photos wrong, so together they beat either one alone: FER validation 0.762 → 0.774, new person 0.767 → 0.780. Live on the webcam, frames that pass the threshold but are wrong dropped from 7.7% to 5.0%. Faces where Vision finds no points fall back to the CNN alone.

> The model reads **facial movement**, not felt emotion. People smile when they are nervous. Don't use it to judge anyone.

### 2. Beatbox sound classifier (a second exercise, not in the app)

| | |
|---|---|
| Task | 0.5-second clip → **clap / hihats / kick / snare** |
| Architecture | AudioCNN, 3 conv blocks on a 64×44 log-mel spectrogram, 23,780 parameters, trained from scratch |
| Training data | Pafras/beatbox-bucket, 4,014 clips from 125 recordings, split **by recording** |
| Honest result | **0.517 ± 0.038** (3 seeds) on 14 amateur beatboxers from the AVP dataset |

The main finding: 0.99 on the bucket's own data only measures the bucket. On other people's voices with a different mic, accuracy falls to ~0.52, just 4 points above always guessing "hihats". Zero-padding also let the model cheat ("short sound followed by silence = hihat"), so clips are padded with noise from the clip itself instead. The biggest problem is data variety, not training settings.

## The experiment story (face model)

Each run changes one variable. Seed and split are fixed, and the test set was opened only once, at the end.

| # | Run | What changed | Result (val) |
|---|---|---|---|
| 0 | TinyCNN | baseline, 2 conv layers | 0.588 |
| 1 | LR too high | lr 1e-2 | stuck, doesn't learn |
| 2 | DeepCNN | 3 conv layers | 0.678 |
| 3 | MobileNet | ImageNet pretrained, 224 px | **0.813**, transfer learning wins by far |
| 4 | ResNet18 | pretrained, 7× bigger | 0.797, bigger ≠ better |
| 5–7 | Small LR, augmentation | lr 1e-4, flip/rotation/brightness | overfitting delayed, plateaus at ~0.81 |
| 8–9 | AdamW, cosine | optimizer, scheduler | wobble gone, 0.827 |
| 10–12 | Weight decay, class weight, dropout | wd 0.05, class weights, dropout 0.5 | 0.837–0.840 |
| 13 | Label smoothing | target 0.925 | confidently wrong 181 → 56 photos |
| 14 | +Sad | 5 classes | 0.770, sad pulls from neutral |
| 15–16 | SGD | optimizer, lr | 0.827, far less memorising |
| 17 | **Clean fine-tune** ⭐ | + 4 recorded people, lr 1e-4 | new person 0.660 → 0.767 |
| 18 | Fine-tune on everyone | + 2 people with doubtful labels | worse: clean labels > more data |
| 19 | Landmark MLP | 76 face points, no pixels | 0.673, 9 points below the CNN |
| 20 | **Fusion** ⭐ | CNN + Landmark MLP | 0.774 FER / 0.780 new person |

## How the app works

1. **Camera → face.** Vision (`VNDetectFaceRectanglesRequest`) finds the face in every frame.
2. **Crop with the "fer" margin.** Vision's box cuts off the eyebrows, so it is widened by 11% of its height upwards and 5.5% of its width on each side, to match a FER2013 crop. This lifted angry from 0.31 to 0.80 on the webcam.
3. **Preprocessing identical to training:** grayscale → **down to 48×48 first** → up to 224×224 → 3 channels → ImageNet normalisation. The 48 step is required: the model learned from blurry 48 px photos, not sharp webcam crops.
4. **Models.** CNN + Landmark MLP in Core ML, ~13 ms per frame.
5. **Anti-jitter.** Majority vote over ~10 frames, a **0.6** confidence threshold, a new expression must hold for 0.8 s, and patterns only switch at the start of a bar.
6. **Beat.** Each expression is a looping pattern, not a single sound. Five genres (Techno, EDM, Breakbeat, R&B, Brazil funk), each with its own tempo and A/B/C variations per expression.
7. **Variation from expression strength.** After calibration (hold a neutral face for 2 seconds), the app measures how far the 76 face points move from that resting face and picks variation A, B or C per bar. Without calibration, variations rotate A → B → A → C.

Every sound comes from the beatbox-bucket dataset, three takes per sound, played in turn. All camera processing happens on the device; no image is sent anywhere.

## Running it

### The app

Needs macOS 27 and Xcode. Open `EmotionBeatbox/EmotionBeatbox.xcodeproj` and press Run. The models and samples are bundled, so the app runs without any Python step. Allow camera access when asked.

### Python

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Datasets are placed by hand (git-ignored): FER2013 in `data/fer2013/{train,test}/<class>/`, beatbox-bucket in `BeatboxAudioDataset/`, AVP v4 in `AVP_Dataset-2/`.

Face model pipeline, in order:

| Step | Script |
|---|---|
| Check the dataset | `scripts/inspect_dataset.py data/fer2013/train` |
| Make the validation split (fixed seed) | `scripts/make_split.py` |
| Train from ImageNet | `scripts/train.py --arch mobilenet --classes 5 ...` |
| Record faces & split by person | `scripts/record_faces.py`, `scripts/make_own_split.py` |
| Fine-tune | `scripts/fine_tune.py` |
| Landmark MLP & fusion | `scripts/extract_landmarks.py`, `scripts/train_landmarks.py`, `scripts/fuse_landmarks.py` |
| Evaluate | `scripts/evaluate.py`, `scripts/predictions.py` |
| Convert to Core ML + check the numbers | `scripts/convert_coreml.py`, `scripts/convert_landmarks.py` |
| Copy into the app | `scripts/sync_app_assets.py` |
| Live webcam test | `scripts/webcam_test.py --fuse` |

`train.py` options: `--arch {tinycnn,deepcnn,resnet18,mobilenet}`, `--lr`, `--epochs`, `--batch-size`, `--seed`, `--augment`, `--optimizer {adam,adamw,sgd}`, `--weight-decay`, `--scheduler {none,cosine}`, `--class-weight`, `--dropout`, `--label-smoothing`, `--classes {4,5}`, `--save`. A new option means a new experiment, with no code edits.

Audio model: `scripts/make_audio_split.py` → `scripts/make_avp_split.py` → `scripts/train_audio.py` → `scripts/evaluate_audio.py`.

## Technical notes

- **Core ML conversion** uses float16, so a gap of ~1e-2 on MobileNetV3's logits against PyTorch is expected (float32 matches to ~1e-7). Confident predictions have margins far wider than that.
- **Python ↔ Swift parity** is checked on 43 "golden crops" from the webcam test: the fused answer matches on 43/43.
- **Vision changes between macOS versions.** After upgrading to macOS 27, face boxes shifted ~9 px. Compare only on the same OS, and re-test live after an upgrade.
- `torch` is pinned to 2.7.0 because it is the newest version coremltools 9.0 has been tested against.

## Limitations

- Sad and neutral are confused most often; sad is the weakest class on new faces.
- Angry is hard to act out: many people make an angry face that is almost flat.
- Fine-tuning used 4 people and was validated on 1, so the new-person numbers are slightly optimistic.
- A camera below the face, dim light and covered faces lower accuracy.
- The audio model is not fit for use beyond this exercise.
