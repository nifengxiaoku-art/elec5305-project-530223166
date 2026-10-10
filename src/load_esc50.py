from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


SELECTED_CATEGORIES = {
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
}


def find_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_esc50_root(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "ESC-50",
        project_root / "data" / "esc50",
        project_root / "data" / "ESC50",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "ESC-50 directory not found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def find_metadata_file(esc50_root: Path) -> Path:
    possible_paths = [
        esc50_root / "META" / "esc50.csv",
        esc50_root / "meta" / "esc50.csv",
        esc50_root / "META" / "ESC-50.csv",
        esc50_root / "meta" / "ESC-50.csv",
    ]
    for path in possible_paths:
        if path.exists():
            return path

    csv_files = sorted(esc50_root.rglob("*.csv"))
    if csv_files:
        return csv_files[0]

    raise FileNotFoundError(f"Could not find ESC-50 metadata CSV under {esc50_root}")


def main() -> None:
    project_root = find_project_root()
    esc50_root = find_esc50_root(project_root)
    metadata_path = find_metadata_file(esc50_root)
    audio_root = esc50_root / "AUDIO"

    if not audio_root.exists():
        raise FileNotFoundError(f"ESC-50 audio directory does not exist: {audio_root}")

    print(f"Project root: {project_root}")
    print(f"ESC-50 root: {esc50_root}")
    print(f"Metadata file: {metadata_path}")
    print(f"Audio dir: {audio_root}")

    filtered_rows = []
    missing_files = []
    class_counts = Counter()
    fold_counts = Counter()

    with metadata_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = (row.get("category") or "").strip().lower()
            filename = (row.get("filename") or "").strip()
            if category not in SELECTED_CATEGORIES:
                continue

            full_path = audio_root / filename
            if not full_path.exists():
                missing_files.append(filename)

            filtered_rows.append(row)
            class_counts[category] += 1
            fold_counts[int(row["fold"])] += 1

    print("\nSelected animal classes:")
    for category in sorted(SELECTED_CATEGORIES):
        print(f"- {category}: {class_counts.get(category, 0)}")

    print("\nOfficial fold counts after filtering:")
    for fold in sorted(fold_counts):
        print(f"- fold {fold}: {fold_counts[fold]}")

    print(f"\nTotal filtered recordings: {len(filtered_rows)}")

    if missing_files:
        print(f"\nWarning: {len(missing_files)} files are missing from AUDIO folder.")
        print("First few missing files:")
        for name in missing_files[:10]:
            print(f"- {name}")

    expected_per_class = 40
    bad_classes = [category for category in sorted(SELECTED_CATEGORIES) if class_counts.get(category, 0) != expected_per_class]
    if bad_classes:
        print("\nWARNING: not all animal classes have 40 recordings:")
        for category in bad_classes:
            print(f"- {category}: {class_counts.get(category, 0)}")
    else:
        print(f"\nAll {len(SELECTED_CATEGORIES)} selected classes have exactly {expected_per_class} recordings.")

    animal_subset_dir = esc50_root / "animal_subset"
    animal_subset_dir.mkdir(exist_ok=True)

    filtered_csv_path = animal_subset_dir / "animal_subset_metadata.csv"
    with filtered_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "fold", "target", "category", "esc10", "src_file", "take"])
        writer.writeheader()
        for row in filtered_rows:
            writer.writerow({
                "filename": row.get("filename", ""),
                "fold": row.get("fold", ""),
                "target": row.get("target", ""),
                "category": row.get("category", ""),
                "esc10": row.get("esc10", ""),
                "src_file": row.get("src_file", ""),
                "take": row.get("take", ""),
            })

    print(f"\nFiltered metadata saved to: {filtered_csv_path}")


if __name__ == "__main__":
    main()
