# Deepfake Detection

Minimal pipeline to download data (manual), preprocess faces, train a model, and run inference.

## Requirements

- Python 3.9+
- macOS Apple Silicon recommended (MPS). CPU also works.
- for cuda, the code is not supported as of now

## Setup

```bash
cd /Users/akashaaprasad/Documents/DeepFake\ Engine_v1/deepfake-detection
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Download

This repo does not automatically download datasets. Put your videos here:

```
data/
  raw/
    real/   # real videos (.mp4/.avi)
    fake/   # deepfake videos (.mp4/.avi)
```

You can run the helper instructions:

```bash
python 1_download_data.py
```

## Preprocess (Extract Faces)

```bash
python 2_preprocess.py
```

This creates:

```
data/
  processed/
    real/<video_name>/*.jpg
    fake/<video_name>/*.jpg
```

## Train

```bash
python 3_train.py
```

The best checkpoint is saved to:

```
models/best_model.pth
```

The training script uses Apple MPS if available; otherwise it falls back to CPU.

## Inference (Single Image)

```bash
python 4_inference.py /full/path/to/image.jpg
```

Example:

```bash
python 4_inference.py /Users/akashaaprasad/Downloads/test4.jpeg
```

If you see an error like `Could not read image`, the file path is wrong or the image cannot be opened.

## Notes

- If you retrain the model, inference will use the new `models/best_model.pth`.
- For faster iteration, use a small set of videos.
