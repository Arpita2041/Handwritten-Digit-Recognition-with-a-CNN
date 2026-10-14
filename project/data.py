"""Offline benchmark data, disjoint image indices, and image input."""
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_data(seed=42):
    digits = load_digits()
    x = digits.images.astype(np.float64)[:, None, :, :] / 16
    y = digits.target
    indices = np.arange(len(y))
    remaining, test = train_test_split(indices, test_size=.15, random_state=seed, stratify=y)
    remaining, calibration = train_test_split(
        remaining, test_size=.10 / .85, random_state=seed + 1, stratify=y[remaining])
    train, validation = train_test_split(
        remaining, test_size=.15 / .75, random_state=seed + 2, stratify=y[remaining])
    return x, y, {"train": train, "validation": validation,
                  "calibration": calibration, "test": test}

def read_image(path, invert=False):
    with Image.open(Path(path)) as image:
        x = np.asarray(image.convert("L").resize((8, 8), Image.Resampling.BILINEAR),
                       dtype=np.float64) / 255
    if invert:
        x = 1 - x
    return x[None, None, :, :]
