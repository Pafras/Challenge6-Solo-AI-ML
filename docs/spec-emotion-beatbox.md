# Emotion-to-Beatbox

## 0. Project Goal

Build a simple interactive macOS application where the user's **facial expressions control beatbox/music**, with an additional **quest/challenge system** that asks the user to perform specific facial-expression sequences to recreate a target beat.

The core experience should be:

```text
User makes facial expression
        ↓
AI recognizes expression
        ↓
Expression becomes musical action
        ↓
Beat is generated
        ↓
User can hear the result
```

The gamified experience extends this into:

```text
Quest
  ↓
Target Beat / Expression Sequence
  ↓
User performs expressions
  ↓
AI recognizes expressions
  ↓
Compare User vs Target
  ↓
Score
  ↓
Quest Complete
```

The project should be achievable within approximately **12 days**.

The priority is a **working, fun, demonstrable prototype**, not state-of-the-art ML.

---

# 1. Core Product Concept

The application is an interactive **facial-expression beatbox game**.

Instead of physically beatboxing with their mouth, the user uses their facial expressions to choose which beat is playing.

**One expression selects one looping pattern, not one sound** (decided 11 Sep). Each emotion maps to a whole pattern that loops on the BPM clock:

```text
😐 Neutral  → K - H - K - H -    (calm)
😄 Happy    → K H H S K H H S    (groove)
😠 Angry    → K K S - K K S S    (heavy)
😮 Surprise → K H S H K S H S    (variation)
```

The exact patterns and tempos are placeholders, settled on Day 10. Sad is not in the model: the face classifier has four classes (angry / happy / neutral / surprise).

A new expression swaps the pattern at the start of the next bar, and only once the prediction is stable and confident. Below the confidence threshold the current pattern keeps playing, so the music never stops or stutters.

Rejected: one expression = one hit (😄 → hi-hat, 😐 → kick, 😠 → snare). Temporal smoothing alone takes about 0.33 s at 30 fps before the user even changes face, so hits could land only every 0.5–1 s — too slow to be a beat — and every flicker in the prediction would sound as a wrong hit.

The user can freely change the beat by changing expressions.

The application can then provide quests such as:

> "Recreate this beat using your face."

Example:

```text
TARGET (one expression per bar):

😄 → 😐 → 😠 → 😄

groove → calm → heavy → groove
```

The user must hold each expression for its bar. The detected per-bar sequence is compared against the target.

---

# 2. Important Scope Decision

The project contains several systems, but they should NOT all be treated as equally important.

## P0 — Must Work

```text
Facial Expression Recognition
        ↓
Emotion
        ↓
Beat Mapping
        ↓
Audio Generation
```

And:

```text
Quest
        ↓
User performs expressions
        ↓
Basic validation
        ↓
Score / completion
```

## P1 — Should Work If Time Allows

```text
Emotion Intensity
ML Beatbox Generator
Advanced scoring
Timing accuracy
Dynamic quest generation
```

## P2 — Future

```text
Facial landmarks
Continuous facial movement features
Human beatbox waveform generation
Neural audio generation
Transformer beat generation
Advanced game mechanics
Multiplayer
```

Do not sacrifice P0 functionality to implement P1/P2 features.

---

# 3. High-Level Architecture

```text
                         ┌─────────────────┐
                         │  QUEST SYSTEM   │
                         └────────┬────────┘
                                  │
                           Target Sequence
                                  │
                                  ▼
Camera → Face Detection → Facial Expression Model
                                  │
                                  ▼
                         Emotion + Intensity
                                  │
                                  ▼
                         Beat Interpretation
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             Audio Generation             Quest Validator
                    │                           │
                    ▼                           ▼
                 Beatbox                 Score / Feedback
```

---

# 4. Main Components

The project consists of six main components:

```text
1. Facial Expression Model
2. Beat Representation
3. Beatbox Generator
4. Audio Engine
5. Quest System
6. macOS Application
```

Each component should be modular.

---

# 5. Facial Expression Model

## Purpose

Recognize the user's facial expression from the camera.

## Input

A cropped face image.

```text
Camera Frame
    ↓
Face Detection
    ↓
Face Crop
    ↓
Facial Expression Model
```

## Output

Use a generic representation:

```json
{
  "emotion": "happy",
  "confidence": 0.94,
  "intensity": 0.82
}
```

The initial model may classify:

```text
angry
disgust
fear
happy
sad
surprise
neutral
```

The exact classes depend on the selected dataset.

---

# 6. Facial Dataset

Use an existing public facial-expression dataset.

Potential datasets:

- FER2013
- RAF-DB
- AffectNet
- CK+
- Other suitable datasets

Selection criteria:

1. Clear labels
2. Easy preprocessing
3. Reasonable size
4. PyTorch compatibility
5. Appropriate license
6. Feasible within 12 days

Do not spend excessive time searching for the perfect dataset.

A "good enough" dataset that enables the complete pipeline is preferred.

---

# 7. Facial Model

Start with a simple model.

Possible architectures:

```text
Small CNN
ResNet18
MobileNet
```

Prefer a lightweight pretrained model if it significantly reduces training time.

Pipeline:

```text
Face Image
    ↓
Resize
    ↓
Normalize
    ↓
Augmentation
    ↓
CNN
    ↓
Emotion Classification
```

The model should prioritize:

- Reasonable accuracy
- Fast inference
- Small model size
- Easy Core ML conversion

---

# 8. Facial Model Evaluation

Evaluate using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Also measure:

- Inference latency
- Real-time prediction stability
- Prediction consistency between frames

The goal is not SOTA performance.

The model only needs to be reliable enough for the interactive experience.

---

# 9. Real-Time Facial Pipeline

Python prototype:

```text
Camera
  ↓
OpenCV / MediaPipe Face Detection
  ↓
Face Crop
  ↓
PyTorch Model
  ↓
Emotion Result
```

Final macOS implementation:

```text
Camera
  ↓
Vision
  ↓
Core ML Facial Expression Model
  ↓
Emotion Result
```

Face detection does not need to be trained from scratch.

---

# 10. Emotion Intensity

Emotion should eventually contain both:

```text
Emotion
+
Intensity
```

Example:

```text
Happy + 0.2
```

means a subtle happy expression.

```text
Happy + 0.9
```

means a strong happy expression.

Intensity can control musical parameters such as:

```text
BPM
Kick Density
Hi-hat Density
Snare Density
Pattern Complexity
Volume
```

This makes the interaction more expressive.

---

# 11. Beat Representation

Do NOT initially generate raw audio using a neural network.

Use symbolic beat tokens.

Initial vocabulary:

```text
K = Kick
H = Hi-hat
S = Snare
- = Silence
```

Example:

```text
K H H S K H H S
```

A beat can be represented as:

```json
{
  "bpm": 120,
  "sequence": ["K", "H", "H", "S", "K", "H", "H", "S"]
}
```

This intermediate representation makes the system easier to train, debug, and deploy.

---

# 12. Beatbox Generator

There should be two possible implementations.

## V1 — Rule-Based Generator

This is the guaranteed MVP.

```text
Emotion + Intensity
        ↓
Musical Parameters
        ↓
Beat Pattern
        ↓
Audio Engine
```

Example:

```text
Happy + High Intensity
        ↓
Fast BPM
High hi-hat density
High kick density
        ↓
K H H K H S H H
```

This should be implemented FIRST.

---

## V2 — ML Beat Generator

If sufficient time remains, replace the rule-based generator with an ML model.

```text
Emotion + Intensity
        ↓
Feature Vector
        ↓
MLP / LSTM / GRU
        ↓
Beat Tokens
```

Example:

```text
Input:
happy, 0.82

Output:
K H H S K H H S
```

Start with MLP/LSTM/GRU.

Only attempt a Transformer if the simpler implementation already works.

---

# 13. Audio Engine

The Audio Engine converts symbolic beat tokens into audio.

Example:

```text
K → kick.wav
H → hihat.wav
S → snare.wav
```

Input:

```text
K H H S
```

Output:

```text
🔊 playable audio
```

The Audio Engine controls:

- BPM
- Timing
- Sample placement
- Volume
- Pattern repetition
- Optional effects

The Audio Engine does not need to be AI.

---

# 14. Main Interactive Mode

The user should be able to freely control the music.

Example:

```text
User holds:  😄 ....... 😄 ....... 😐 ....... 😠
Bar:         1          2          3          4
Playing:     groove     groove     calm       heavy
```

The beat changes live: once a new expression is stable and confident, its pattern takes over at the start of the next bar. A flicker inside a bar changes nothing.

This mode demonstrates the core AI interaction.

---

# 15. Quest System

The Quest System adds a game layer.

A quest defines a target sequence the user needs to reproduce.

Example:

```text
Quest #01
Happy Groove

Target (one expression per bar):
😄 → 😐 → 😠 → 😄
```

Equivalent beat (one pattern per bar):

```text
groove → calm → heavy → groove
```

The user holds each expression for its bar:

```text
😄 → 😐 → 😠 → 😄
```

The system records the expression detected in each bar and compares that sequence with the target.

---

# 16. Quest Structure

Represent quests using structured data.

Example:

```json
{
  "id": "quest_001",
  "name": "Happy Groove",
  "difficulty": "easy",
  "bpm": 100,
  "steps": [
    { "emotion": "happy",   "pattern": "groove" },
    { "emotion": "neutral", "pattern": "calm" },
    { "emotion": "angry",   "pattern": "heavy" },
    { "emotion": "happy",   "pattern": "groove" }
  ]
}
```

Each step lasts one bar. `pattern` names an entry in the emotion → pattern lookup table, so the quest file never repeats the tokens.

This should remain simple and easy to modify.

---

# 17. Quest Gameplay

Example:

```text
QUEST

Happy Groove

Perform (one bar each):

😄        😐        😠        😄
 ↓         ↓         ↓         ↓
groove    calm      heavy     groove

[ START ]
```

After starting:

```text
Timeline:

  bar 1     bar 2     bar 3     bar 4
   😄        😐        😠        😄
   |─────────|─────────|─────────|─────────|
```

The user holds each expression through its bar and hears that bar's pattern.

---

# 18. Quest Validation

The Quest Validator compares:

```text
Target Expression
vs
Detected Expression
```

and optionally:

```text
Target Timing
vs
User Timing
```

Example:

```text
Target: Happy
User:   Happy
Result: ✓
```

If timing is included:

```text
Target: 1.00 sec
User:   1.08 sec

Timing Accuracy: 92%
```

---

# 19. Scoring

Initial scoring can be simple.

Possible factors:

```text
Expression Accuracy
+
Timing Accuracy
+
Sequence Accuracy
```

Example:

```text
Expression Accuracy: 95%
Timing Accuracy:     88%
Sequence Accuracy:   100%

TOTAL SCORE: 94
```

Use simple scoring initially.

Do not build a complicated game economy or progression system.

---

# 20. Feedback

The app should provide immediate feedback.

Examples:

```text
PERFECT! 🔥
```

```text
GOOD!
```

```text
MISS
```

At the end:

```text
QUEST COMPLETE

Score: 94%

🔥 Excellent!
```

The feedback is important because it makes the ML system feel interactive.

---

# 21. Quest Difficulty

## Easy

Short sequence:

```text
😄 → 😐 → 😠
```

## Medium

Longer sequence:

```text
😄 → 😐 → 😠 → 😮 → 😄
```

## Hard

Faster sequence:

```text
😄 😐 😠 😮 😠 😄 😐 😮
```

## Future: Intensity Challenges

```text
😄 30%
😄 80%
😠 40%
😠 90%
```

The user must reproduce both expression and intensity.

Intensity-based quests are P1/P2.

---

# 22. Quest Types

Possible future quest types:

### Type A — Expression Sequence

```text
😄 → 😐 → 😠 → 😄
```

### Type B — Beat Reproduction

User listens to:

```text
K H H S
```

and tries to recreate it using expressions.

### Type C — Timing Challenge

User must perform expressions at specific moments.

### Type D — Intensity Challenge

User must perform the same emotion at different intensities.

### Type E — Free Style

No fixed sequence.

The AI generates music based on the user's expressions.

Only Type A should be required for the first version.

---

# 23. Core Game Loop

The complete experience should eventually be:

```text
Choose Quest
      ↓
Show Target
      ↓
Start Countdown
      ↓
Camera Tracking
      ↓
Facial Expression Recognition
      ↓
Convert Expression → Beat
      ↓
Compare with Target
      ↓
Generate Audio
      ↓
Calculate Score
      ↓
Show Result
```

---

# 24. Technical Architecture

## Python / ML

Responsible for:

```text
Dataset
Training
Evaluation
Model Export
Experimentation
```

## macOS

Responsible for:

```text
Camera
Face Detection
Model Inference
Quest UI
Audio
Scoring
User Interaction
```

Recommended stack:

```text
Python
PyTorch
torchvision
NumPy
OpenCV
librosa
soundfile
```

macOS:

```text
SwiftUI
Vision
Core ML
AVFoundation
```

---

# 25. Model Deployment

Training:

```text
PyTorch
    ↓
Trained Model
    ↓
Export
    ↓
Core ML
    ↓
macOS App
```

The final macOS application should ideally not depend on a Python runtime.

---

# 26. Repository Structure

```text
emotion-beatbox/
│
├── datasets/
│   ├── facial_expression/
│   └── beatbox/
│
├── models/
│   ├── facial_expression/
│   └── beatbox_generator/
│
├── training/
│   ├── train_emotion.py
│   ├── train_beatbox.py
│   └── evaluate.py
│
├── preprocessing/
│   ├── image_preprocessing.py
│   └── audio_preprocessing.py
│
├── inference/
│   ├── emotion_inference.py
│   └── beatbox_inference.py
│
├── audio/
│   ├── samples/
│   │   ├── kick.wav
│   │   ├── snare.wav
│   │   └── hihat.wav
│   └── audio_engine.py
│
├── realtime/
│   ├── camera.py
│   └── pipeline.py
│
├── quest/
│   ├── quest_data.json
│   ├── quest_validator.py
│   └── scoring.py
│
├── configs/
│   └── config.yaml
│
├── notebooks/
│
├── tests/
│
├── scripts/
│   ├── train.sh
│   ├── evaluate.sh
│   └── export_coreml.sh
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 27. Developer Environment

Required:

```text
macOS
Xcode
Python 3.x
Git
GitHub
PyTorch
```

Recommended Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Dependencies should be defined in:

```text
requirements.txt
```

Do not rely on globally installed Python packages.

---

# 28. Configuration

Keep important settings outside the source code.

Example:

```yaml
emotion_model:
  image_size: 224
  num_classes: 7
  learning_rate: 0.001
  batch_size: 32
  epochs: 20

audio:
  default_bpm: 120
  sample_rate: 44100

beat:
  sequence_length: 16
```

This makes experiments easier to reproduce.

---

# 29. Reproducibility

Training should use fixed random seeds when appropriate.

Save:

```text
Model checkpoint
Training configuration
Validation metrics
Dataset information
```

Example:

```text
models/facial_expression/
├── best_model.pth
├── config.yaml
└── metrics.json
```

---

# 30. Data Management

Do not commit large datasets into Git.

Instead:

```text
README
   ↓
Dataset Download Instructions
   ↓
Local datasets/
```

The repository should contain only the necessary metadata/scripts.

Also document the license of:

- Facial dataset
- Audio samples
- Any pretrained model

---

# 31. Privacy

The application should process camera data locally.

Preferred architecture:

```text
Camera
   ↓
Local Processing
   ↓
AI
   ↓
Audio
```

No cloud inference is required.

The app should request appropriate macOS camera/microphone permissions.

Do not upload facial images to a server.

---

# 32. 12-Day Implementation Plan

## Day 1

### Facial Dataset + Environment

- Select dataset
- Set up Python
- Set up PyTorch
- Implement DataLoader
- Verify dataset

---

## Day 2

### Facial Preprocessing

- Resize
- Normalize
- Augmentation
- Train/validation/test split

---

## Day 3

### Facial Model

- Implement baseline CNN
- Train model
- Save checkpoint
- Evaluate

Target:

```text
Image → Emotion
```

---

## Day 4

### Model Improvement

- Test pretrained model
- Tune basic parameters
- Generate confusion matrix
- Select final model

---

## Day 5

### Real-Time Facial Recognition

Implement:

```text
Camera
→ Face Detection
→ Face Crop
→ Emotion Model
→ Emotion UI
```

---

## Day 6

### Beat Representation + Audio

- Define tokens
- Collect/create audio samples
- Implement audio engine
- Create rule-based generator

Target:

```text
Emotion → Beat → Audio
```

---

## Day 7

### Beatbox ML Model

If time permits:

```text
Emotion + Intensity
→ ML model
→ Beat Sequence
```

Otherwise improve rule-based generation.

---

## Day 8

### Quest System

Implement:

- Quest data structure
- Target sequences
- Quest state
- Basic validator

---

## Day 9

### Quest Scoring

Implement:

```text
Expression matching
Sequence matching
Basic timing
Score
Feedback
```

---

## Day 10

### Full Integration

Connect:

```text
Camera
→ Face
→ Emotion
→ Beat
→ Audio
→ Quest Validator
→ Score
```

---

## Day 11

### macOS UI

Build simple SwiftUI interface.

Screens:

```text
Home
Quest Selection
Quest Gameplay
Result
```

Keep UI simple.

---

## Day 12

### Polish + Demo

- Fix bugs
- Optimize latency
- Improve audio transitions
- Test expressions
- Test quests
- Polish UI
- Prepare demo

---

# 33. MVP User Flow

The minimum complete user journey:

```text
OPEN APP
   ↓
HOME
   ↓
SELECT QUEST
   ↓
"Happy Groove"
   ↓
SHOW TARGET
   ↓
START
   ↓
USER MAKES EXPRESSIONS
   ↓
AI RECOGNIZES EXPRESSIONS
   ↓
BEAT PLAYS
   ↓
SYSTEM CHECKS SEQUENCE
   ↓
SCORE
   ↓
QUEST COMPLETE
```

---

# 34. MVP Example

Quest:

```text
Happy Groove

Target:

😄 → 😐 → 😠 → 😄
```

Mapping (one pattern per bar):

```text
😄 → groove   K H H S K H H S
😐 → calm     K - H - K - H -
😠 → heavy    K K S - K K S S
😄 → groove   K H H S K H H S
```

User holds, one bar each:

```text
😄 → 😐 → 😠 → 😄
```

System detects, per bar:

```text
Happy ✓
Neutral ✓
Angry ✓
Happy ✓
```

Result:

```text
Sequence: 100%
Timing:   91%

SCORE: 96

QUEST COMPLETE 🔥
```

---

# 35. Important Implementation Constraint

The system should NOT require perfect facial recognition.

Real-time facial predictions can fluctuate:

```text
happy
happy
neutral
happy
happy
```

Even when the user maintains the same expression.

Therefore, implement basic temporal smoothing.

For example:

```text
Recent predictions:

happy
happy
happy
neutral
happy

→ Final state: happy
```

Possible approaches:

- Majority voting
- Moving average
- Confidence threshold
- Prediction debounce

Keep this simple.

---

# 36. Beat Transition

Avoid restarting the entire audio every time the expression changes.

Instead:

```text
Current Beat
     ↓
Expression Change
     ↓
Schedule New Pattern
     ↓
Next Musical Step
```

This should produce smoother musical transitions.

Real-time audio synchronization is more important than complex AI.

---

# 37. Architecture Principle

The system should be designed around interfaces.

For example:

```text
FacialExpressionRecognizer
        ↓
FacialExpressionResult
        ↓
BeatGenerator
        ↓
BeatSequence
        ↓
AudioEngine
```

The BeatGenerator should not care how the facial model was implemented.

It should only receive something like:

```json
{
  "emotion": "happy",
  "intensity": 0.82
}
```

This allows the facial model to be replaced later.

---

# 38. Beat Generator Interface

Conceptually:

```text
generateBeat(
    emotion,
    intensity
) → BeatSequence
```

Example:

```text
generateBeat("happy", 0.82)

→

{
  bpm: 128,
  sequence: ["K", "H", "H", "S", "K", "H", "H", "S"]
}
```

---

# 39. Quest Validator Interface

Conceptually:

```text
validate(
    targetSequence,
    detectedSequence
) → Score
```

Example:

```text
Target:
H K S H

Detected:
H K S H

→ 100%
```

---

# 40. What NOT to Build Initially

Do not spend time on:

```text
❌ Custom face detector
❌ Neural raw-audio generation
❌ Large Transformer
❌ Complex multiplayer
❌ Cloud backend
❌ User accounts
❌ Database
❌ Social features
❌ Complex progression system
❌ Huge audio library
❌ Advanced 3D graphics
```

The core experience is more important.

---

# 41. Success Criteria

The project is successful if a user can:

1. Open the macOS application.
2. Select a quest.
3. See/hear the target.
4. Perform facial expressions.
5. Have the AI recognize the expressions.
6. Hear corresponding beatbox sounds.
7. Complete a sequence.
8. Receive a score.

Most importantly:

> **Changing the user's facial expression should visibly and audibly change the music.**

And in Quest Mode:

> **The user should feel like they are performing a beat using their face.**

---

# 42. Future Expansion

## Facial Interaction

```text
Emotion
   ↓
Emotion + Intensity
   ↓
Facial Landmarks
   ↓
Facial Movement
   ↓
Continuous Expression Control
```

## Music Generation

```text
Rule-Based Beat
   ↓
ML Beat Sequence
   ↓
Human Beatbox Samples
   ↓
Neural Audio Generation
```

## Game

```text
Simple Quest
   ↓
Difficulty
   ↓
Combo
   ↓
Streak
   ↓
Achievements
   ↓
Procedural Quests
```

---

# 43. Potential "Wow" Moment

The demo should ideally contain this moment:

```text
Presenter:

"Watch this."

😄

→ upbeat hi-hat

😠

→ heavy kick/snare

😢

→ slower beat

😮

→ unusual rhythm
```

Then:

> "Now let's try a challenge."

The app gives:

```text
😄 → 😐 → 😠 → 😄
```

The user performs it.

The app responds:

```text
PERFECT! 🔥

96%
```

This demonstrates both:

```text
AI
+
Interaction
+
Gamification
```

rather than showing only an ML classification screen.

---

# 44. Final Product Definition

**Emotion-to-Beatbox** is a real-time macOS interactive music game where facial expressions become musical controls, and users can complete quests by performing specific expression sequences to recreate target beats.

Core loop:

```text
FACE
 ↓
EMOTION
 ↓
BEAT
 ↓
AUDIO
 ↓
QUEST
 ↓
SCORE
```

The project should remain technically simple enough to complete in 12 days while leaving a clear path toward more advanced facial interaction, ML beat generation, and neural audio generation.