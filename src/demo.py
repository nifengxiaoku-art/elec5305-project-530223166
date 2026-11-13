"""Portable code demonstration: real classifier fitting, validation and figures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory, for example runs/demo_01")
    parser.add_argument("--data-root", default="data/ESC-50")
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--cache", default="cache/embeddings")
    parser.add_argument("--checkpoint", help="Optional local Microsoft CLAP checkpoint")
    parser.add_argument("--offline", action="store_true",
                        help="Use only previously downloaded model/text assets")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous results; choose a new --output path")
    print("Running the frozen-CLAP clean study with real classifier fitting.", flush=True)
    print("Missing audio embeddings are extracted; compatible features are reused.", flush=True)
    command = [sys.executable, "-u", "-m", "src.run_study",
               "--config", args.config, "--data-root", args.data_root,
               "--cache", args.cache, "--output", str(args.output), "--clean-only"]
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    if args.offline:
        command.append("--offline")
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, "-m", "src.analyze_study", str(args.output)], check=True)

    runs = pd.read_csv(args.output/"metrics/runs.csv")
    examples = runs[(~runs.pilot) & (runs.method == "clean") &
                    (runs.test_fold == 5) & (runs.n == 4)].sort_values("seed")
    example = examples.iloc[0] if len(examples) else runs[~runs.pilot].iloc[0]
    searches = pd.read_csv(args.output/"metrics/validation_search.csv")
    search = searches[searches.run_id == example.run_id].sort_values("C")
    print(f"\nC selection example: {example.run_id}", flush=True)
    print("C is inverse regularization strength: smaller C means stronger regularization.")
    print(search[["C", "validation_macro_f1"]].to_string(index=False))
    print(f"Selected C = {example.C:g}: highest validation Macro-F1; smaller C wins ties.")
    print(f"Held-out test Macro-F1 for this one run = {example.macro_f1:.4f}")
    print("The five-fold mean is a separate number in analysis/summary.csv.")
    print(f"Training originals: {int(example.train_originals)}; "
          f"additional validation labels: {int(example.validation_labels)}")
    summary = pd.read_csv(args.output/"analysis/summary.csv")
    print("\nFive-fold conditions:")
    print(summary[summary.complete_five_folds][
        ["method", "n", "macro_f1_mean", "macro_f1_fold_sd", "accuracy_mean"]
    ].to_string(index=False))
    print(f"\nPredictions: {args.output/'predictions/few_shot.csv'}")
    print(f"Learning curve: {args.output/'analysis/learning_curve.png'}")
    print("This demo uses clean training/test audio; the full study adds matched training noise.")
    manifest = {
        "scope": "Fixed clean protocol; actual validation-selected classifier fitting",
        "example_run": example.run_id, "selected_C": float(example.C),
        "example_test_macro_f1": float(example.macro_f1),
        "classifier_fits": len(runs), "summary": "analysis/summary.csv",
        "validation_search": "metrics/validation_search.csv",
        "predictions": "predictions/few_shot.csv",
    }
    (args.output/"demo.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
