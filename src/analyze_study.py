"""Recompute predictions, audit boundaries and create descriptive research exhibits."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score
from .dataset import CATEGORIES, sha256
from .evaluate import metrics
from .extract_embeddings import write_json


def figure_save(fig, root, name):
    fig.savefig(root/f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(root/f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run
    if not (root/"completed.json").exists():
        raise ValueError("Run must finish before final analysis")
    destination = root/"analysis"
    destination.mkdir(exist_ok=False)
    zero = pd.read_csv(root/"predictions"/"zero_shot.csv", dtype={"src_file": str})
    pred = pd.read_csv(root/"predictions"/"few_shot.csv", dtype={"src_file": str})
    runs = pd.read_csv(root/"metrics"/"runs.csv")
    selected = pd.read_csv(root/"manifests"/"selected_training.csv", dtype={"src_file": str})
    config = json.loads((root/"config.json").read_text())
    tuning = pd.read_csv(root/"metrics"/"validation_search.csv")
    stored_zero = json.loads((root/"metrics"/"zero_shot.json").read_text())
    audit = {"prediction_runs_checked": 0, "test_predictions_checked": len(pred),
             "source_disjoint_splits_checked": 0, "validation_selections_checked": 0}
    if len(zero) != 400 or zero.filename.nunique() != 400:
        raise ValueError("Zero-shot must predict each of 400 clips exactly once")
    zero_metrics = []
    for fold, group in zero.groupby("fold"):
        result = metrics(group.true_label, group.predicted_label)
        if len(group) != 80:
            raise ValueError("Expected 80 test clips")
        for metric in ["macro_f1", "accuracy"]:
            np.testing.assert_allclose(result[metric], stored_zero[str(fold)][metric], atol=1e-12)
        np.testing.assert_array_equal(result["confusion_matrix"], stored_zero[str(fold)]["confusion_matrix"])
        zero_metrics.append({"test_fold": fold, "method": "zero_shot", "n": 0,
                             "macro_f1": result["macro_f1"], "accuracy": result["accuracy"]})
    for fold in range(1, 6):
        roles = {name: pd.read_csv(root/"manifests"/f"fold{fold}_{name}.csv", dtype={"src_file": str})
                 for name in ["train", "validation", "test"]}
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            if set(roles[a].src_file) & set(roles[b].src_file):
                raise ValueError("Source leakage")
        audit["source_disjoint_splits_checked"] += 1
        test_lookup = roles["test"].set_index("filename")
        for _, record in runs[runs.test_fold == fold].iterrows():
            group = pred[pred.run_id == record.run_id]
            if len(group) != 80 or set(group.filename) != set(test_lookup.index):
                raise ValueError("Incomplete/incorrect held-out predictions")
            if group.filename.duplicated().any():
                raise ValueError("Duplicate test predictions")
            if any(test_lookup.loc[r.filename, "category"] != r.true_label for r in group.itertuples()):
                raise ValueError("Incorrect ground truth")
            result = metrics(group.true_label, group.predicted_label)
            for metric in ["macro_f1", "accuracy"]:
                np.testing.assert_allclose(result[metric], record[metric], atol=1e-12)
            for category in CATEGORIES:
                np.testing.assert_allclose(result["recall"][category], record[f"recall_{category}"], atol=1e-12)
            chosen = selected[selected.run_id == record.run_id]
            if chosen.filename.duplicated().any() or not set(chosen.filename).issubset(roles["train"].filename):
                raise ValueError("Invalid selected training originals")
            if not (chosen.groupby("category").size().reindex(CATEGORIES) == record.n).all():
                raise ValueError("Incorrect labels-per-class count")
            search = tuning[tuning.run_id == record.run_id].sort_values("C")
            expected = search[search.validation_macro_f1 >= search.validation_macro_f1.max()-1e-12].iloc[0].C
            if record.C != expected or set(search.C) != set(config["C_grid"]):
                raise ValueError("Classifier C does not match validation selection")
            audit["validation_selections_checked"] += 1
            audit["prediction_runs_checked"] += 1
    actual = runs[~runs.pilot].copy()
    fold_means = actual.groupby(["method", "n", "test_fold"], as_index=False)[["macro_f1", "accuracy"]].mean()
    within_sd = actual.groupby(["method", "n", "test_fold"], as_index=False).macro_f1.std().rename(columns={"macro_f1":"seed_sd_macro_f1"})
    fold_means = fold_means.merge(within_sd, on=["method", "n", "test_fold"])
    fold_means.to_csv(destination/"fold_seed_means.csv", index=False)
    population = pd.concat([fold_means, pd.DataFrame(zero_metrics)], ignore_index=True)
    summary = population.groupby(["method", "n"]).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_fold_sd=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_fold_sd=("accuracy", "std"), folds=("test_fold", "nunique")).reset_index()
    summary["complete_five_folds"] = summary.folds == 5
    summary.to_csv(destination/"summary.csv", index=False)
    pd.DataFrame(zero_metrics).to_csv(destination/"zero_shot_folds.csv", index=False)
    pooled = {"zero_shot": metrics(zero.true_label, zero.predicted_label)}
    # Pool each seed separately: repeated predictions of one original are not new test clips.
    for (method, n, seed), group in pred[~pred.pilot].groupby(["method", "n", "seed"]):
        if group.test_fold.nunique() == 5:
            pooled[f"{method}_n{n}_seed{seed}"] = metrics(group.true_label, group.predicted_label)
    write_json(destination/"pooled_per_seed.json", pooled)
    all_recall = []
    for _, record in actual.iterrows():
        for category in CATEGORIES:
            all_recall.append({"method": record.method, "n": int(record.n), "test_fold": int(record.test_fold),
                               "seed": int(record.seed), "category": category, "recall": record[f"recall_{category}"]})
    for category in CATEGORIES:
        all_recall.append({"method":"zero_shot", "n":0, "test_fold":0, "seed":0, "category":category,
                           "recall": pooled["zero_shot"]["recall"][category]})
    pd.DataFrame(all_recall).to_csv(destination/"class_recall.csv", index=False)
    errors = zero[zero.true_label != zero.predicted_label].groupby(["true_label", "predicted_label"]).size().sort_values(ascending=False)
    errors.rename("count").to_csv(destination/"zero_shot_errors.csv")
    aug = actual[actual.method == "noise"]
    if len(aug):
        pairs = aug.merge(actual[actual.method == "clean"], on=["test_fold", "n", "seed"], suffixes=("_noise", "_clean"), validate="one_to_one")
        for pair in pairs.itertuples():
            a = selected[selected.run_id == pair.run_id_noise].filename
            b = selected[selected.run_id == pair.run_id_clean].filename
            if set(a) != set(b):
                raise ValueError("Unmatched augmentation originals")
        pairs["delta_macro_f1"] = pairs.macro_f1_noise - pairs.macro_f1_clean
        pairs["delta_accuracy"] = pairs.accuracy_noise - pairs.accuracy_clean
        pairs.to_csv(destination/"paired_deltas.csv", index=False)
        paired_folds = pairs.groupby(["n", "test_fold"], as_index=False)[["delta_macro_f1", "delta_accuracy"]].mean()
        paired_folds.to_csv(destination/"paired_fold_means.csv", index=False)
        paired_summary = paired_folds.groupby("n").delta_macro_f1.agg(["mean", "std", "min", "max"])
        paired_summary.to_csv(destination/"paired_summary.csv")
        snr = pd.read_csv(root/"metrics"/"noise_snr.csv")
        if not np.allclose(snr.measured_snr_db, config["snr_db"], atol=1e-4):
            raise ValueError("Waveform SNR mismatch")
        audit["matched_pairs_checked"] = len(pairs)
        audit["snr_waveforms_checked"] = len(snr)
        audit["snr_min"] = float(snr.measured_snr_db.min())
        audit["snr_max"] = float(snr.measured_snr_db.max())
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        for n, group in paired_folds.groupby("n"):
            x = np.full(len(group), int(n), dtype=float)
            offsets = np.linspace(-.6, .6, len(group))
            ax.scatter(x+offsets, group.delta_macro_f1*100, label=f"n={n}: five fold means")
            ax.errorbar(n, group.delta_macro_f1.mean()*100, yerr=group.delta_macro_f1.std()*100,
                        fmt="ks", capsize=5)
        ax.axhline(0, color="gray", lw=1)
        ax.set(xticks=config["augmentation_n"], xlabel="Labelled original clips per class",
               ylabel="Noise − clean Macro-F1 (percentage points)")
        ax.legend(fontsize=8)
        figure_save(fig, destination, "augmentation_deltas")
    # No line connects zero-shot with supervised models: n=0 is a different system.
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    curve = summary[(summary.method == "clean") & summary.complete_five_folds]
    ax.errorbar(curve.n, curve.macro_f1_mean, yerr=curve.macro_f1_fold_sd, fmt="o-", capsize=4,
                label="Frozen CLAP + LR: mean ± fold SD")
    zs = summary[summary.method == "zero_shot"].iloc[0]
    ax.axhline(zs.macro_f1_mean, color="tab:orange", linestyle="--", label="Fixed-prompt zero-shot mean")
    ax.fill_between([1,16], zs.macro_f1_mean-zs.macro_f1_fold_sd, zs.macro_f1_mean+zs.macro_f1_fold_sd,
                    color="tab:orange", alpha=.12, label="Zero-shot ± fold SD")
    for fold, g in fold_means[(fold_means.method == "clean") & (fold_means.n < 24)].groupby("test_fold"):
        ax.plot(g.n, g.macro_f1, color="gray", alpha=.25, lw=.7)
    ax.set(xticks=[1,2,4,8,16], xlabel="Labelled original training clips per class", ylabel="Macro-F1", ylim=(0,1.02))
    ax.legend(fontsize=8, loc="lower right")
    figure_save(fig, destination, "learning_curve")
    fig, axes = plt.subplots(1, 2, figsize=(11,4.8), constrained_layout=True)
    cm = np.asarray(pooled["zero_shot"]["confusion_matrix"])
    for ax, values, title in zip(axes, [cm, cm/cm.sum(1,keepdims=True)], ["Pooled zero-shot counts (400 clips)", "Recall-normalized zero-shot"]):
        image = ax.imshow(values, cmap="Blues")
        for i in range(10):
            for j in range(10):
                if values[i,j]:
                    ax.text(j,i, f"{values[i,j]:.0f}" if ax == axes[0] else f"{values[i,j]:.2f}", ha="center",va="center",fontsize=7,
                            color="white" if values[i,j]>values.max()/2 else "black")
        ax.set(xticks=range(10),yticks=range(10),xticklabels=CATEGORIES,yticklabels=CATEGORIES,
               xlabel="Predicted category",ylabel="True category",title=title)
        ax.tick_params(axis="x",rotation=65,labelsize=8)
        ax.tick_params(axis="y",labelsize=8)
        fig.colorbar(image, ax=ax, shrink=.7)
    figure_save(fig, destination, "zero_shot_confusion")
    legacy = Path("results/zero_shot_fold5_metrics.json")
    audit["legacy_fold5_match"] = (
        bool(np.isclose(stored_zero["5"]["macro_f1"],
                        json.loads(legacy.read_text())["macro_f1"]))
        if legacy.is_file() else None
    )
    audit["prediction_sha256"] = {p.name:sha256(p) for p in (root/"predictions").glob("*.csv")}
    audit["status"] = "PASS: metric, selection, boundary and SNR consistency checks; see methodological limitations"
    write_json(destination/"audit.json", audit)
    print(summary.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
