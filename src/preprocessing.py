"""Data acquisition, preprocessing and feature extraction."""
import io
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import (BATCH_SIZE, CLASS_NAMES, DATA_DIR, IMAGE_EXTENSIONS,
                     IMG_SIZE, SEED, TEST_DIR, TRAIN_DIR)

KAGGLE_DATASET = "iarunava/cell-images-for-detecting-malaria"


# --------------------------------------------------------------------------
# 1. Data acquisition
# --------------------------------------------------------------------------
def download_dataset():
    """Download the NIH malaria cell-image dataset from Kaggle (needs kagglehub).

    Returns the folder that contains the 'Parasitized' and 'Uninfected' folders.
    """
    import kagglehub

    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    return find_class_root(path)


def find_class_root(path):
    """Locate the directory holding the Parasitized/Uninfected sub-folders."""
    path = Path(path)
    for p in [path, *path.rglob("*")]:
        if p.is_dir() and (p / "Parasitized").is_dir() and (p / "Uninfected").is_dir():
            return p
    raise FileNotFoundError(f"Could not find Parasitized/Uninfected folders under {path}")


def _list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def prepare_split(raw_dir, per_class=None, test_fraction=0.15, seed=SEED, clean=True):
    """Copy images from the raw download into data/train and data/test.

    per_class: cap on images per class (None = use everything). Useful to keep the
    GitHub repo small and training fast.
    """
    raw_dir = find_class_root(raw_dir)
    rng = random.Random(seed)
    if clean:
        for split in (TRAIN_DIR, TEST_DIR):
            for cls in CLASS_NAMES:
                d = split / cls
                if d.exists():
                    shutil.rmtree(d)
    summary = {}
    for cls in CLASS_NAMES:
        files = _list_images(raw_dir / cls)
        rng.shuffle(files)
        if per_class:
            files = files[:per_class]
        n_test = int(len(files) * test_fraction)
        splits = {"test": files[:n_test], "train": files[n_test:]}
        for split_name, items in splits.items():
            out = (TEST_DIR if split_name == "test" else TRAIN_DIR) / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, out / f.name)
        summary[cls] = {k: len(v) for k, v in splits.items()}
    return summary


# --------------------------------------------------------------------------
# 2. Data processing
# --------------------------------------------------------------------------
def load_image_array(source):
    """Load an image (path, bytes or file-like) as RGB float32 array (H, W, 3), 0-255.

    Scaling to [-1, 1] happens *inside* the model (Rescaling layer), so training and
    serving always share exactly the same preprocessing.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    img = Image.open(source).convert("RGB").resize(IMG_SIZE)
    return np.asarray(img, dtype=np.float32)


def make_datasets(train_dir=TRAIN_DIR, test_dir=TEST_DIR, batch_size=BATCH_SIZE,
                  val_split=0.15, seed=SEED):
    """Build tf.data pipelines: (train, validation, test)."""
    import tensorflow as tf

    common = dict(labels="inferred", label_mode="binary", class_names=CLASS_NAMES,
                  image_size=IMG_SIZE, batch_size=batch_size, seed=seed)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=val_split, subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=val_split, subset="validation", **common)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, shuffle=False, **{**common, "seed": None})
    auto = tf.data.AUTOTUNE
    return (train_ds.prefetch(auto), val_ds.prefetch(auto), test_ds.prefetch(auto))


def count_images(root):
    return {cls: len(_list_images(Path(root) / cls)) if (Path(root) / cls).exists() else 0
            for cls in CLASS_NAMES}


# --------------------------------------------------------------------------
# 3. Feature extraction for visualisations / interpretation
# --------------------------------------------------------------------------
def image_features(source):
    """Interpretable image-level features used in the data-insight visualisations."""
    a = load_image_array(source)
    gray = a.mean(axis=2)
    hsv = np.asarray(Image.fromarray(a.astype(np.uint8)).convert("HSV"), dtype=np.float32)
    cell = gray > 20                      # cells are cropped on a black background
    cell_px = max(int(cell.sum()), 1)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    return {
        "brightness": float(gray[cell].mean()) if cell.any() else 0.0,
        "green_mean": float(a[..., 1][cell].mean()) if cell.any() else 0.0,
        "saturation": float(hsv[..., 1][cell].mean()) if cell.any() else 0.0,
        "cell_area": float(cell.mean()),
        "dark_stain_ratio": float(((gray <= 110) & cell).sum() / cell_px),
        "edge_strength": float((gx + gy) / 2),
    }


FEATURE_NAMES = ["brightness", "green_mean", "saturation",
                 "cell_area", "dark_stain_ratio", "edge_strength"]


def dataset_features(data_dir=TRAIN_DIR, per_class=300, seed=SEED):
    """Sample images per class and return a DataFrame of features + label."""
    rng = random.Random(seed)
    rows = []
    for cls in CLASS_NAMES:
        files = _list_images(Path(data_dir) / cls) if (Path(data_dir) / cls).exists() else []
        rng.shuffle(files)
        for f in files[:per_class]:
            try:
                rows.append({"label": cls, **image_features(f)})
            except Exception:
                continue
    return pd.DataFrame(rows)


def sample_images(data_dir=TRAIN_DIR, n=4, seed=SEED):
    """Return {class: [paths]} for plotting example cells."""
    rng = random.Random(seed)
    out = {}
    for cls in CLASS_NAMES:
        files = _list_images(Path(data_dir) / cls) if (Path(data_dir) / cls).exists() else []
        rng.shuffle(files)
        out[cls] = files[:n]
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download and split the malaria dataset")
    ap.add_argument("--raw-dir", help="Existing folder with Parasitized/Uninfected (skips download)")
    ap.add_argument("--per-class", type=int, default=None, help="Cap images per class")
    args = ap.parse_args()
    raw = args.raw_dir or download_dataset()
    print(prepare_split(raw, per_class=args.per_class))
    print("train:", count_images(TRAIN_DIR), "test:", count_images(TEST_DIR))
