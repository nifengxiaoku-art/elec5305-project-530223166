"""Reproduce every experiment and analysis from code, data and the fixed model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New experiment directory")
    parser.add_argument("--data-root", default="data/ESC-50")
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--cache", default="cache/embeddings")
    parser.add_argument("--checkpoint", help="Optional local checkpoint; configured hash is checked")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory to preserve previous experiments")
    executed = []

    def run(module, arguments):
        command = [sys.executable, "-u", "-m", module, *map(str, arguments)]
        print(f"\n=== {module} ===", flush=True)
        subprocess.run(command, check=True)
        executed.append(module)

    study = ["--output", args.output, "--data-root", args.data_root,
             "--config", args.config, "--cache", args.cache]
    if args.checkpoint:
        study += ["--checkpoint", args.checkpoint]
    if args.offline:
        study += ["--offline"]
    run("src.run_study", study)
    run("src.analyze_study", [args.output])
    run("src.acoustic_analysis", [args.output, "--data-root", args.data_root])
    run("src.noise_diagnostics", [args.output, "--data-root", args.data_root])
    run("src.sensitivity", [args.output, "--data-root", args.data_root, "--cache", args.cache])
    interpretation = args.output/"interpretation"
    run("src.deepen_analysis", [args.output, "--output", interpretation])
    run("src.plot_confusion", [args.output/"predictions/zero_shot.csv",
                              "--output-prefix", interpretation/"zero_shot_confusion_readable"])
    verification = [args.output, "--data-root", args.data_root, "--cache", args.cache]
    if args.offline:
        verification += ["--offline"]
    run("src.verify_study", verification)
    result = {"status": "PASS", "steps_completed": executed,
              "metrics": "analysis/summary.csv", "paired_noise_effects": "analysis/paired_summary.csv",
              "figures": ["analysis", "acoustics", "interpretation"],
              "verification": "verification.json"}
    (args.output/"reproduction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nComplete. Results and generated figures: {args.output}", flush=True)


if __name__ == "__main__":
    main()
