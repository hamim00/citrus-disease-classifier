# scripts/split_dataset.py
import os
import shutil
import random
import json
from typing import List

# -------- CONFIG --------
SOURCE_DIR = r"data_sources/processed_master"     # master dataset (one folder per class)
DEST_DIR   = r"data/carambola/processed"          # where train/val/test will be created
SPLIT_RATIOS = (0.7, 0.15, 0.15)                  # train, val, test
COPY_MODE = "copy"  # "copy" or "move"
RANDOM_SEED = 42
EXTS = (".jpg", ".jpeg", ".png")
# ------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_classes(src: str) -> List[str]:
    classes = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
    classes = sorted(classes)  # stable order for class_names.json
    if not classes:
        raise RuntimeError(f"No class folders found in: {src}")
    return classes

def split_files(files: List[str], ratios):
    total = len(files)
    t = int(total * ratios[0])
    v = int(total * ratios[1])
    # test is remainder
    return files[:t], files[t:t+v], files[t+v:]

def copy_or_move(src_path, dst_path):
    if COPY_MODE == "move":
        shutil.move(src_path, dst_path)
    else:
        shutil.copy2(src_path, dst_path)

def main():
    random.seed(RANDOM_SEED)

    classes = list_classes(SOURCE_DIR)
    print("Classes:", classes)

    # write class names for the app
    ensure_dir("weights")
    with open("weights/class_names.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print(f"Saved weights/class_names.json with {len(classes)} classes.")

    # prepare split dirs
    for split in ["train", "val", "test"]:
        ensure_dir(os.path.join(DEST_DIR, split))

    totals = {"train":0, "val":0, "test":0}

    for cname in classes:
        csrc = os.path.join(SOURCE_DIR, cname)
        imgs = [f for f in os.listdir(csrc) if f.lower().endswith(EXTS)]
        if not imgs:
            print(f"WARNING: no images found in class '{cname}'")
            continue

        random.shuffle(imgs)
        train_files, val_files, test_files = split_files(imgs, SPLIT_RATIOS)

        # make class dirs in each split
        for split in ["train", "val", "test"]:
            ensure_dir(os.path.join(DEST_DIR, split, cname))

        # copy/move
        for f in train_files:
            copy_or_move(os.path.join(csrc, f),
                         os.path.join(DEST_DIR, "train", cname, f))
        for f in val_files:
            copy_or_move(os.path.join(csrc, f),
                         os.path.join(DEST_DIR, "val", cname, f))
        for f in test_files:
            copy_or_move(os.path.join(csrc, f),
                         os.path.join(DEST_DIR, "test", cname, f))

        print(f"[{cname}]  train={len(train_files)}  val={len(val_files)}  test={len(test_files)}")
        totals["train"] += len(train_files)
        totals["val"]   += len(val_files)
        totals["test"]  += len(test_files)

    print("\nDone!")
    print("Totals:", totals)
    print("Train dir:", os.path.abspath(os.path.join(DEST_DIR, "train")))
    print("Val dir:  ", os.path.abspath(os.path.join(DEST_DIR, "val")))
    print("Test dir: ", os.path.abspath(os.path.join(DEST_DIR, "test")))
    if COPY_MODE == "move":
        print("\nNOTE: Files were MOVED out of", os.path.abspath(SOURCE_DIR))
    else:
        print("\nNOTE: Files were COPIED; master data remains in", os.path.abspath(SOURCE_DIR))

if __name__ == "__main__":
    main()
