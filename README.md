# Citrus Disease Classifier

A PyTorch-based image classifier for citrus disease detection using CNNs and ResNet backbones, with simple training and inference utilities.

> **Dataset:** We do not distribute the dataset in this repo. Download it from Mendeley:  
> https://data.mendeley.com/datasets/f35jp46gms/1  
> Place it under `data/` following the folder layout below (or pass custom paths via CLI).

---

## Project Structure

citrus-disease-classifier/
├── models/
│ ├── init.py
│ ├── custom_cnn.py
│ ├── registry.py
│ └── scnn.py
├── utils/
│ ├── init.py
│ ├── inference.py
│ ├── io.py
│ └── transforms.py
├── scripts/
│ └── split_dataset.py
├── xai/
│ ├── init.py
│ ├── cam_utils.py
│ └── lime_utils.py
├── weights/
│ └── class_names.json # small metadata kept in Git
├── app.py
├── train.py
├── requirements.txt
└── README.md


> **Ignored:** `data/`, `data_sources/`, `*.pt` weights, `__pycache__/`, `.venv/`, etc. See `.gitignore`.

---

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


### 2) Dataset

Download from Mendeley: https://data.mendeley.com/datasets/f35jp46gms/1

After download/extract, place images under:

data/
  Citrus Nutrient Deficiency/
  Healthy Fruits/
  Healthy Leaves/
  Insect Hole leaves/
  Unhealthy Fruits/

### 3) (Optional) Split Train/Val/Test

If your dataset is not already split:

python scripts/split_dataset.py \

  --src_dir data \
  --out_dir data_splits \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42



Training

Basic example (adjust paths, batch size, epochs as needed):
python train.py \
  --data_dir data_splits/train \
  --val_dir data_splits/val \
  --model resnet18 \
  --epochs 25 \
  --batch_size 32 \
  --lr 3e-4 \
  --seed 42 \
  --save_dir weights


Reproducibility

Use the provided requirements.txt.

Set seeds for Python & PyTorch when training:

import random, os, numpy as np, torch
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
