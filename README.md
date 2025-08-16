# Citrus Disease Classifier

A PyTorch-based image classifier for citrus disease detection using CNNs and ResNet backbones, with simple training and inference utilities.

> **Dataset:** We do not distribute the dataset in this repo. Download it from Mendeley:  
> https://data.mendeley.com/datasets/f35jp46gms/1  
> Place it under `data/` following the folder layout below (or pass custom paths via CLI).

---

## Features

- Modular model registry (custom CNN + ResNet backbones)
- Clear train/inference scripts
- Optional dataset splitter
- XAI helpers (e.g., CAM/LIME utilities)
- Reproducible seeds and deterministic training settings

---

## Project Structure

```text
citrus-disease-classifier/
├── models/
│   ├── __init__.py
│   ├── custom_cnn.py
│   ├── registry.py
│   └── scnn.py
├── utils/
│   ├── __init__.py
│   ├── inference.py
│   ├── io.py
│   └── transforms.py
├── scripts/
│   └── split_dataset.py
├── xai/
│   ├── __init__.py
│   ├── cam_utils.py
│   └── lime_utils.py
├── weights/
│   └── class_names.json    # small metadata kept in Git
├── app.py
├── train.py
├── requirements.txt
└── README.md

```
Ignored: data/, data_sources/, *.pt weights, __pycache__/, .venv/, etc. See .gitignore.

## Getting Started
### 1) Environment
```bash
# Python 3.11 recommended
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```
### 2) Dataset
Download from Mendeley: https://data.mendeley.com/datasets/f35jp46gms/1

After download/extract, place images under:

```text

data/
  Citrus Nutrient Deficiency/
  Healthy Fruits/
  Healthy Leaves/
  Insect Hole leaves/
  Unhealthy Fruits/
```
If your local layout differs, you can pass paths via CLI flags (see Training/Inference).

### 3) (Optional) Split Train/Val/Test
If your dataset is not already split:

```bash

python scripts/split_dataset.py \
  --src_dir data \
  --out_dir data_splits \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42
```
This will produce:

```text
data_splits/
  train/
  val/
  test/
Training
```
Basic example (adjust paths, batch size, epochs as needed):

```bash
python train.py \
  --data_dir data_splits/train \
  --val_dir data_splits/val \
  --model resnet18 \
  --epochs 25 \
  --batch_size 32 \
  --lr 3e-4 \
  --seed 42 \
  --save_dir weights
```
Trained weights (*.pt) are not committed to Git by default. Use GitHub Releases, cloud storage, or Git LFS if you need to share them.

## Inference
Single image example:

```bash

python -m utils.inference \
  --weights path/to/model_best.pt \
  --image path/to/sample.jpg \
  --class_map weights/class_names.json
  ```
Batch folder inference (if supported by utils/inference.py):

```bash

python -m utils.inference \
  --weights path/to/model_best.pt \
  --image_dir path/to/images \
  --class_map weights/class_names.json \
  --out_csv predictions.csv
```
## Reproducibility
Use the provided requirements.txt.

Set seeds for Python & PyTorch when training (ensure your train.py calls a function like this):

```python

import os, random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```
Run training with the same --seed value for consistent splits.

Notes
Weights: Store large models via GitHub Releases, Google Drive, or Git LFS. Update links in this README if you publish them.

Transforms: See utils/transforms.py for augmentations used in training/inference.

Model Registry: models/registry.py collects model builders (e.g., custom_cnn.py, scnn.py, resnet18 if defined).

XAI: See xai/ utilities for CAM/LIME usage (add short usage examples here if needed).