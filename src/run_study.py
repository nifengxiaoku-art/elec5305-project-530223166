"""Run strict dataset audit, frozen extraction, pilot and full matched study."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
import pandas as pd
from .dataset import CATEGORIES, load_dataset, split_rows, sample_rows, sha256
from .evaluate import metrics
from .extract_embeddings import FrozenCLAP, write_json, normalized_matrix
from .few_shot import fit_classifier


def dataset_audit(full, rows, root):
    split_details = []
    for fold in range(1, 6):
        train, val, test = split_rows(rows, fold)
        split_details.append({"test_fold": fold, "validation_fold": 5 if fold == 1 else fold-1,
                              "train_count": len(train), "validation_count": len(val), "test_count": len(test),
                              "train_per_class": train.groupby("category").size().to_dict(),
                              "excluded_train": rows[~rows.fold.isin([fold, 5 if fold == 1 else fold-1]) & ~rows.index.isin(train.index)].filename.tolist(),
                              "excluded_validation": rows[(rows.fold == (5 if fold == 1 else fold-1)) & ~rows.index.isin(val.index)].filename.tolist()})
    cross = rows.groupby("src_file").filter(lambda g: g.fold.nunique() > 1)
    return {"metadata_sha256": sha256(Path(root)/"meta"/"esc50.csv"), "total_rows": len(full),
            "selected_rows": len(rows), "class_fold_counts": pd.crosstab(rows.category, rows.fold).to_dict(),
            "crossfold_sources": cross.drop(columns="path").to_dict("records"), "splits": split_details}


def git_revision():
    """Record a checkout revision when available; downloaded ZIPs have no HEAD."""
    try:
        project = Path(__file__).resolve().parents[1]
        checkout = subprocess.check_output(
            ["git", "-C", str(project), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        if Path(checkout).resolve() != project:
            return None
        return subprocess.check_output(
            ["git", "-C", str(project), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/ESC-50")
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--output", required=True, help="New directory; existing outputs are never replaced")
    parser.add_argument("--cache", default="cache/embeddings")
    parser.add_argument("--checkpoint")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--clean-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config["categories"] != CATEGORIES or len(config["prompts"]) != 10:
        raise ValueError("Class/prompt alignment must match the declared task")
    full, rows = load_dataset(args.data_root)
    output.mkdir(parents=True)
    for name in ["metrics", "predictions", "figures", "manifests"]:
        (output/name).mkdir()
    write_json(output/"config.json", config)
    write_json(output/"metrics"/"dataset_audit.json", dataset_audit(full, rows, args.data_root))
    rows.drop(columns="path").to_csv(output/"manifests"/"animal_metadata.csv", index=False)
    for fold in range(1, 6):
        for role, part in zip(["train", "validation", "test"], split_rows(rows, fold)):
            part.drop(columns="path").to_csv(output/"manifests"/f"fold{fold}_{role}.csv", index=False)
    if args.audit_only:
        print("Dataset and all five source-disjoint splits validated", flush=True)
        return
    write_json(output/"environment.json", {"python": platform.python_version(), "platform": platform.platform(),
               "utc_started": datetime.now(timezone.utc).isoformat(),
               "git_head": git_revision(),
               "source_sha256": {str(p): sha256(p) for p in sorted(Path("src").glob("*.py"))},
               "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}})
    checkpoint = args.checkpoint
    if checkpoint is None and config.get("checkpoint_revision"):
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download(
            repo_id=config["checkpoint_repo"], filename=config["checkpoint_filename"],
            revision=config["checkpoint_revision"], local_files_only=args.offline
        )
    extractor = FrozenCLAP(config, checkpoint, args.offline)
    if config.get("checkpoint_sha256") and extractor.manifest["checkpoint_sha256"] != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 differs from the fixed research configuration")
    write_json(output/"model.json", extractor.manifest)
    clean, cache_index, clean_dir = extractor.extract(rows, args.cache)
    cache_index.to_csv(output/"manifests"/"clean_cache.csv", index=False)
    text = extractor.text_embeddings()
    np.save(output/"manifests"/"text_embeddings.npy", text, allow_pickle=False)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    cosine = normalized_matrix(clean, rows.index) @ text.T
    zero = rows.drop(columns="path").copy()
    zero["true_label"] = zero.category
    zero["predicted_label"] = np.array(CATEGORIES)[cosine.argmax(1)]
    for i, category in enumerate(CATEGORIES):
        zero[f"cosine_{category}"] = cosine[:, i]
    zero.to_csv(output/"predictions"/"zero_shot.csv", index=False)
    write_json(output/"metrics"/"zero_shot.json", {str(f): metrics(g.true_label, g.predicted_label)
               for f, g in zero.groupby("fold")})
    runs, predictions, tuning, selections, skipped = [], [], [], [], []

    def run_cell(fold, n, seed, method, noisy=None, pilot=False):
        train, val, test = split_rows(rows, fold)
        selected = sample_rows(train, n, seed)
        x = normalized_matrix(clean, selected.index)
        y = selected.category.to_numpy()
        weights = None
        if method == "noise":
            x = np.concatenate([x, normalized_matrix(noisy, selected.index)])
            y = np.tile(y, 2)
            weights = np.full(len(y), 0.5)
        classifier, c, validation = fit_classifier(x, y, normalized_matrix(clean, val.index),
                                                 val.category.to_numpy(), config["C_grid"], weights)
        predicted = classifier.predict(normalized_matrix(clean, test.index))
        result = metrics(test.category, predicted)
        run_id = f"{'pilot_' if pilot else ''}{method}_f{fold}_n{n}_s{seed}"
        identity = {"run_id": run_id, "method": method, "test_fold": fold,
                    "validation_fold": 5 if fold == 1 else fold-1, "seed": seed, "n": n, "pilot": pilot}
        record = {**identity, "C": c, "train_originals": len(selected),
                  "train_sources": selected.src_file.nunique(), "validation_labels": len(val),
                  "macro_f1": result["macro_f1"], "accuracy": result["accuracy"],
                  **{f"recall_{k}": v for k,v in result["recall"].items()}}
        runs.append(record)
        for item in validation:
            tuning.append({**identity, **item})
        for _, row in selected.iterrows():
            selections.append({**identity, "filename": row.filename, "src_file": row.src_file,
                               "category": row.category, "fold": int(row.fold)})
        for (_, row), label in zip(test.iterrows(), predicted):
            predictions.append({**identity, "filename": row.filename, "src_file": row.src_file,
                                "true_label": row.category, "predicted_label": label})
        print(f"{run_id}: Macro-F1={result['macro_f1']:.4f}, C={c}", flush=True)

    # Required pilot first, without using its score to change the frozen design.
    for seed in config["seeds"]:
        run_cell(5, 4, seed, "clean", pilot=True)
    pd.DataFrame(runs).to_csv(output/"metrics"/"pilot.csv", index=False)
    for fold in range(1, 6):
        train, _, _ = split_rows(rows, fold)
        for n in config["n_values"]:
            if train.groupby("category").size().min() < n:
                skipped.append({"test_fold": fold, "n": n, "reason": "Insufficient source-isolated originals per class"})
                continue
            for seed in config["seeds"]:
                run_cell(fold, n, seed, "clean")
    # Persist the completed clean study before secondary encoding.
    def save():
        pd.DataFrame(runs).to_csv(output/"metrics"/"runs.csv", index=False)
        pd.DataFrame(predictions).to_csv(output/"predictions"/"few_shot.csv", index=False)
        pd.DataFrame(tuning).to_csv(output/"metrics"/"validation_search.csv", index=False)
        pd.DataFrame(selections).to_csv(output/"manifests"/"selected_training.csv", index=False)
        write_json(output/"metrics"/"skipped_cells.json", skipped)
    save()
    if not args.clean_only:
        union = set()
        for fold in range(1, 6):
            train, _, _ = split_rows(rows, fold)
            for n in config["augmentation_n"]:
                for seed in config["seeds"]:
                    union.update(sample_rows(train, n, seed).index)
        noisy, noisy_index, noisy_dir = extractor.extract(rows.loc[sorted(union)], args.cache, noisy=True)
        noisy_index.to_csv(output/"manifests"/"noisy_cache.csv", index=False)
        snr_records = []
        for _, row in noisy_index.iterrows():
            details = json.loads((noisy_dir/(row.cache_key+".json")).read_text())
            snr_records.append({"filename": row.filename, **details})
        pd.DataFrame(snr_records).to_csv(output/"metrics"/"noise_snr.csv", index=False)
        for fold in range(1, 6):
            for n in config["augmentation_n"]:
                for seed in config["seeds"]:
                    run_cell(fold, n, seed, "noise", noisy=noisy)
        save()
    write_json(output/"completed.json", {"utc_finished": datetime.now(timezone.utc).isoformat(),
               "clean_cache": str(clean_dir), "noise_completed": not args.clean_only,
               "runs_excluding_pilot": sum(not r["pilot"] for r in runs)})


if __name__ == "__main__":
    main()
