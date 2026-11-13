"""Strict ESC-50 validation and source-isolated official fold rotation."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf

CATEGORIES = ["dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_dataset(root, check_audio=True):
    root = Path(root)
    metadata = root / "meta" / "esc50.csv"
    if not metadata.is_file():
        raise FileNotFoundError(f"Official meta/esc50.csv required: {metadata}")
    full = pd.read_csv(metadata, dtype={"src_file": str})
    required = {"filename", "category", "fold", "src_file", "take", "target", "esc10"}
    if not required.issubset(full.columns) or len(full) != 2000:
        raise ValueError("Expected official 2,000-row ESC-50 metadata and columns")
    if full.filename.nunique() != 2000 or set(full.fold) != set(range(1, 6)):
        raise ValueError("Duplicate filenames or invalid official folds")
    if full[list(required)].isna().any().any():
        raise ValueError("Missing metadata values")
    rows = full[full.category.isin(CATEGORIES)].sort_values("filename").reset_index(drop=True)
    counts = pd.crosstab(rows.category, rows.fold).reindex(CATEGORIES)
    if len(rows) != 400 or counts.shape != (10, 5) or not (counts == 8).all().all():
        raise ValueError("Animal subset must contain 40/class and 8/class/fold")
    if not all(int(f.split('-')[0]) == fold for f, fold in zip(rows.filename, rows.fold)):
        raise ValueError("Filename and metadata folds disagree")
    rows["path"] = [str(root / "audio" / f) for f in rows.filename]
    if check_audio:
        for path in rows.path:
            info = sf.info(path)
            if (info.samplerate, info.channels, info.frames) != (44100, 1, 220500):
                raise ValueError(f"Unexpected ESC-50 audio format: {path}: {info}")
    return full, rows


def split_rows(rows, test_fold):
    if test_fold not in range(1, 6):
        raise ValueError("test_fold must be 1..5")
    val_fold = 5 if test_fold == 1 else test_fold - 1
    test = rows[rows.fold == test_fold]
    original_val = rows[rows.fold == val_fold]
    val = original_val[~original_val.src_file.isin(test.src_file)]
    reserved = set(test.src_file) | set(original_val.src_file)
    train = rows[~rows.fold.isin([test_fold, val_fold]) & ~rows.src_file.isin(reserved)]
    for left, right in [(train, val), (train, test), (val, test)]:
        if set(left.src_file) & set(right.src_file) or set(left.filename) & set(right.filename):
            raise ValueError("Source or filename leakage")
    return train, val, test


def sample_rows(train, n, seed):
    rng = np.random.default_rng(seed)
    indices = []
    for category in CATEGORIES:
        available = train[train.category == category].sort_values("filename").index.to_numpy()
        if len(available) < n:
            raise ValueError(f"n={n} infeasible for {category}: {len(available)} eligible")
        indices.extend(rng.permutation(available)[:n])
    return train.loc[indices]
