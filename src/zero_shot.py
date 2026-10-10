from __future__ import annotations

import csv
import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from msclap import CLAP
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def _compat_torchaudio_load(path, *args, **kwargs):
    """Use soundfile backend when torchaudio's default loader is unavailable."""
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim == 1:
        waveform = waveform.reshape(1, -1)
    else:
        waveform = waveform.T
    return torch.tensor(waveform, dtype=torch.float32), sample_rate


torchaudio.load = _compat_torchaudio_load


TEST_FOLD = 5
SELECTED_CATEGORIES = [
    "dog",
    "rooster",
    "pig",
    "cow",
    "frog",
    "cat",
    "hen",
    "insects",
    "sheep",
    "crow",
]
PROMPTS = {
    "dog": "sound of a dog barking",
    "rooster": "sound of a rooster",
    "pig": "sound of a pig",
    "cow": "sound of a cow",
    "frog": "sound of a frog",
    "cat": "sound of a cat",
    "hen": "sound of a hen",
    "insects": "sound of flying insects",
    "sheep": "sound of a sheep",
    "crow": "sound of a crow",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_test_rows() -> list[dict]:
    metadata_path = project_root() / "data" / "ESC-50" / "META" / "esc50.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    rows = []
    with metadata_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = (row.get("category") or "").strip().lower()
            if category not in SELECTED_CATEGORIES:
                continue
            if int(row.get("fold", 0)) != TEST_FOLD:
                continue
            rows.append(row)

    return rows


def main() -> None:
    root = project_root()
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    test_rows = load_test_rows()
    if not test_rows:
        raise ValueError(f"No ESC-50 samples found for fold {TEST_FOLD} in the selected animal classes.")

    audio_dir = root / "data" / "ESC-50" / "AUDIO"
    file_paths = []
    true_labels = []
    for row in test_rows:
        filename = row["filename"].strip()
        full_path = audio_dir / filename
        if not full_path.exists():
            raise FileNotFoundError(f"Missing audio for {filename}")
        file_paths.append(str(full_path))
        true_labels.append((row["category"] or "").strip().lower())

    print(f"Preparing zero-shot test for fold {TEST_FOLD}.")
    print(f"Number of test samples: {len(file_paths)}")

    model = CLAP()
    text_queries = [PROMPTS[c] for c in SELECTED_CATEGORIES]
    text_embeddings = model.get_text_embeddings(text_queries)
    audio_embeddings = model.get_audio_embeddings(file_paths)
    similarities = model.compute_similarity(audio_embeddings, text_embeddings)

    pred_indices = similarities.argmax(dim=1).tolist()
    predicted_labels = [SELECTED_CATEGORIES[idx] for idx in pred_indices]

    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(true_labels, predicted_labels, labels=SELECTED_CATEGORIES, average="macro")
    per_class_recall = recall_score(
        true_labels,
        predicted_labels,
        labels=SELECTED_CATEGORIES,
        average=None,
        zero_division=0,
    )
    cm = confusion_matrix(true_labels, predicted_labels, labels=SELECTED_CATEGORIES)

    print("\nZero-shot classification results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Per-class recall:")
    for category, recall_value in zip(SELECTED_CATEGORIES, per_class_recall):
        print(f"- {category}: {recall_value:.4f}")
    print("\nConfusion matrix:")
    print(cm)

    metrics = {
        "fold": TEST_FOLD,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_recall": {category: float(value) for category, value in zip(SELECTED_CATEGORIES, per_class_recall)},
        "labels": SELECTED_CATEGORIES,
        "confusion_matrix": cm.tolist(),
    }

    with (results_dir / "zero_shot_fold5_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    with (results_dir / "zero_shot_fold5_confusion_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + SELECTED_CATEGORIES)
        for true_idx, row in enumerate(cm):
            writer.writerow([SELECTED_CATEGORIES[true_idx]] + row.tolist())

    print(f"\nSaved metrics to: {results_dir / 'zero_shot_fold5_metrics.json'}")
    print(f"Saved confusion matrix to: {results_dir / 'zero_shot_fold5_confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
