# ASL Sign Language Recognition System

A real-time American Sign Language (ASL) interpreter that recognizes fingerspelled letters and digits via webcam and converts them into spoken English using text-to-speech.

## Overview

This system combines two separate recognition pipelines with automatic mode-switching:
- **Letters** — MediaPipe hand-landmark extraction + Random Forest classifier
- **Digits** — Fine-tuned YOLOv8n-cls (image-based classification)

A confidence-margin threshold automatically decides, frame by frame, whether the detected sign is a letter or a digit — no manual switching required.

## Key Results

| Module | Approach | Real-World Webcam Accuracy |
|---|---|---|
| Letters | MediaPipe landmarks + Random Forest | **80.8%** (up from 42.3% before fixes) |
| Letters (comparison) | Raw-pixel CNN | 7.7% |
| Digits | YOLOv8n-cls | **87.9%** |
| Digits (comparison) | MobileNetV1/V2/V3, EfficientNet-B0/V2 | 41–71% |

A significant part of this project involved diagnosing and fixing a **train-deployment domain gap** — models scored 98-100% on validation data but dropped to 8-17% on real webcam input. This was resolved through targeted real-world data augmentation and hyperparameter tuning.

## Architecture

```
Camera Feed
    │
    ▼
MediaPipe Hand Detection
    │
    ├── Letter Model (Random Forest on normalized landmarks)
    └── Digit Model (YOLOv8n-cls on cropped hand region)
    │
    ▼
Confidence-Based Mode Selection
    │
    ▼
Stability-Gated Sign Confirmation (~1 sec hold)
    │
    ▼
Sentence Assembly (space gesture / keyboard delete)
    │
    ▼
Text-to-Speech Output
```

## Project Structure

```
├── main.py                  # Entry point — runs the live pipeline
├── pipeline.py              # Mode-switching, stability tracking, sentence assembly
├── letter_inference.py      # MediaPipe landmark extraction + Random Forest prediction
├── digit_inference.py       # Hand cropping + YOLO digit prediction
├── tts.py                   # Text-to-speech wrapper
├── models/
│   ├── best_model.pkl       # Trained Random Forest (letters)
│   └── yolo_digits.pt       # Fine-tuned YOLOv8n-cls (digits)
├── letters.ipynb            # Letter model training & evaluation
├── digits.ipynb             # Digit model comparison (7 architectures)
└── combined_pipeline.ipynb  # Integration and testing notebook
```


## Setup

```bash
uv sync
```

## Usage

```bash
uv run python main.py
```

**Controls:**
- Sign a letter or digit and hold steady for ~1 second to confirm
- `space` gesture — end current word
- `b` — delete last character
- `s` — speak the current sentence aloud
- `q` — quit and print final sentence

## Known Limitations

- Reduced accuracy on visually similar letter clusters (A/M/N/S — differ mainly by thumb position; R/U)
- Digit dataset is small (57 images/class); real-world robustness depends on added webcam training data
- Delete gesture was tested but caused confusion with letters (M/N) in landmark space — replaced with keyboard input
- Currently trained primarily on one signing hand

## Model Comparison Methodology

Seven pretrained CNN architectures were fine-tuned and benchmarked for digit classification (MobileNetV1/V2/V3-Small/V3-Large, EfficientNet-B0/V2-Small, YOLOv8n-cls), evaluated on validation accuracy, dataset test accuracy, and — critically — real webcam accuracy, since validation accuracy alone proved a poor predictor of real-world performance across the project.

A separate comparison of transfer-learning strategies (feature extraction vs. partial fine-tuning vs. full fine-tuning) on MobileNetV2 showed real-world accuracy scaling from 32.8% to 65.5% with the extent of network adaptation.

## Tech Stack

Python · MediaPipe · scikit-learn · PyTorch · torchvision · Ultralytics YOLO · OpenCV · pyttsx3