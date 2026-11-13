"""Exploratory interpretation of saved predictions; no fitting or test tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .dataset import CATEGORIES, sha256
from .extract_embeddings import write_json
from .analyze_study import figure_save


def matched_transitions(left, right, keys):
    """Align identical held-out cases and classify corrections and new errors."""
    fields = keys + ["true_label", "predicted_label"]
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("Duplicate comparison keys")
    paired = left[fields].merge(right[fields], on=keys, how="outer",
                                suffixes=("_before", "_after"), validate="one_to_one", indicator=True)
    if not (paired._merge == "both").all():
        raise ValueError("Comparison has unmatched held-out cases")
    if not (paired.true_label_before == paired.true_label_after).all():
        raise ValueError("Comparison ground truth differs")
    labels = set(paired.true_label_before) | set(paired.predicted_label_before) | set(paired.predicted_label_after)
    if not labels.issubset(CATEGORIES):
        raise ValueError("Unknown class label")
    paired = paired.drop(columns=["_merge", "true_label_after"]).rename(columns={"true_label_before":"true_label"})
    before = paired.true_label == paired.predicted_label_before
    after = paired.true_label == paired.predicted_label_after
    paired["transition"] = np.select([~before & after, before & ~after, ~before & ~after],
                                      ["repaired", "harmed", "persistent_error"], default="stable_correct")
    return paired


def manual_macro_f1(true, predicted):
    """Independent fixed-order F1 calculation for artifact verification."""
    true, predicted = np.asarray(true), np.asarray(predicted)
    scores = []
    for category in CATEGORIES:
        tp = np.sum((true == category) & (predicted == category))
        denominator = np.sum(true == category) + np.sum(predicted == category)
        scores.append(2*tp/denominator if denominator else 0.)
    return float(np.mean(scores))


def leave_one_fold_out(values):
    """Descriptive concentration diagnostic, not confidence intervals or a test."""
    values = pd.Series(values, dtype=float)
    if len(values) != 5 or not np.isfinite(values).all():
        raise ValueError("Five finite paired fold means required")
    return pd.Series({fold: float(values.drop(fold).mean()) for fold in values.index})


def transition_summary(group):
    result = {"prediction_instances":len(group), "unique_test_clips":group.filename.nunique()}
    for name in ["repaired", "harmed", "persistent_error", "stable_correct"]:
        part = group[group.transition == name]
        result[f"{name}_instances"] = len(part)
        result[f"{name}_unique_clips"] = part.filename.nunique()
    result["net_correct_instances"] = result["repaired_instances"]-result["harmed_instances"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="A new interpretation-output directory")
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError("Preserve old outputs; choose a new analysis directory")
    out.mkdir(parents=True)
    run = args.run
    config = json.loads((run/"config.json").read_text())
    zero = pd.read_csv(run/"predictions"/"zero_shot.csv", dtype={"src_file":str}).rename(columns={"fold":"test_fold"})
    primary = pd.read_csv(run/"predictions"/"few_shot.csv", dtype={"src_file":str})
    primary = primary[~primary.pilot]
    fixed = pd.read_csv(run/"sensitivity_fixed_c1"/"predictions.csv")
    frames = []
    keys = ["filename", "test_fold"]
    for protocol, predictions in [("validation_selected", primary), ("fixed_C1_exploratory", fixed)]:
        for seed in config["seeds"]:
            for n in [1,2,4,8,16]:
                clean = predictions[(predictions.method == "clean") & (predictions.seed == seed) & (predictions.n == n)]
                pair = matched_transitions(zero, clean, keys)
                if len(pair) != 400:
                    raise ValueError("Expected every original test clip once per seed")
                frames.append(pair.assign(protocol=protocol, comparison="zero_to_clean", n=n, seed=seed))
            for n in [4,16]:
                clean = predictions[(predictions.method == "clean") & (predictions.seed == seed) & (predictions.n == n)]
                noise = predictions[(predictions.method == "noise") & (predictions.seed == seed) & (predictions.n == n)]
                frames.append(matched_transitions(clean, noise, keys).assign(
                    protocol=protocol, comparison="clean_to_noise", n=n, seed=seed))
    transitions = pd.concat(frames, ignore_index=True).merge(
        zero[["filename", "src_file"]], on="filename", validate="many_to_one")
    transitions.to_csv(out/"prediction_transitions.csv", index=False)
    summaries, class_rows, clip_rows = [], [], []
    dimensions = ["protocol", "comparison", "n"]
    for identity, group in transitions.groupby(dimensions):
        identity = dict(zip(dimensions, identity))
        summary = {**identity, **transition_summary(group)}
        summary["unique_affected_sources"] = group[group.transition.isin(["repaired", "harmed"])].src_file.nunique()
        summaries.append(summary)
        for category, part in group.groupby("true_label"):
            class_rows.append({**identity, "category":category, **transition_summary(part)})
        for filename, part in group.groupby("filename"):
            row = {**identity, "filename":filename, "src_file":part.src_file.iloc[0],
                   "test_fold":int(part.test_fold.iloc[0]), "true_label":part.true_label.iloc[0]}
            for name in ["repaired", "harmed", "persistent_error", "stable_correct"]:
                row[f"{name}_seeds"] = int((part.transition == name).sum())
            clip_rows.append(row)
    summaries = pd.DataFrame(summaries)
    summaries.to_csv(out/"transition_summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out/"class_transitions.csv", index=False)
    clips = pd.DataFrame(clip_rows)
    clips.to_csv(out/"clip_transition_counts.csv", index=False)
    cases = []
    for n in [4,16]:
        base = clips[(clips.protocol == "validation_selected") & (clips.comparison == "clean_to_noise") & (clips.n == n)]
        for name, condition in [("repaired_all_seeds",base.repaired_seeds == 3),
                                ("repaired_any_seed",base.repaired_seeds > 0),
                                ("harmed_any_seed",base.harmed_seeds > 0)]:
            part = base[condition].sort_values("filename")
            if len(part):
                cases.append({"selection_rule":name, **part.iloc[0].to_dict()})
    pd.DataFrame(cases).to_csv(out/"selected_error_cases.csv", index=False)
    fold_rows, checks = [], 0
    for protocol, predictions in [("validation_selected", primary), ("fixed_C1_exploratory", fixed)]:
        stored = pd.read_csv(run/("metrics/runs.csv" if protocol == "validation_selected" else "sensitivity_fixed_c1/runs.csv"))
        for (method,n,fold,seed), group in predictions.groupby(["method","n","test_fold","seed"]):
            result = manual_macro_f1(group.true_label, group.predicted_label)
            expected = stored[(stored.method == method) & (stored.n == n) & (stored.test_fold == fold) & (stored.seed == seed)]
            if "pilot" in expected:
                expected = expected[~expected.pilot]
            if len(expected) != 1 or not np.isclose(result, expected.iloc[0].macro_f1, atol=1e-12):
                raise ValueError("Independent F1 recomputation differs from stored run")
            fold_rows.append({"protocol":protocol, "method":method, "n":n, "test_fold":fold, "seed":seed, "macro_f1":result})
            checks += 1
    scores = pd.DataFrame(fold_rows)
    folds = scores.groupby(["protocol","method","n","test_fold"], as_index=False).macro_f1.mean()
    score_summary = folds.groupby(["protocol","method","n"]).macro_f1.agg(["mean","std","count"]).reset_index()
    score_summary.to_csv(out/"protocol_comparison.csv", index=False)
    sensitivity, paired = [], []
    for protocol in scores.protocol.unique():
        for n in [4,16]:
            data = scores[(scores.protocol == protocol) & (scores.n == n)]
            data = data.pivot(index=["test_fold","seed"], columns="method", values="macro_f1")
            data["delta"] = data.noise-data.clean
            means = data.delta.groupby("test_fold").mean()
            for omitted, value in leave_one_fold_out(means).items():
                sensitivity.append({"protocol":protocol, "n":n, "omitted_test_fold":omitted,
                                    "remaining_four_fold_mean_delta":value})
            paired.append({"protocol":protocol, "n":n, "delta_mean":float(means.mean()),
                           "delta_fold_sd":float(means.std(ddof=1)),
                           "loo_min":float(leave_one_fold_out(means).min()),
                           "loo_max":float(leave_one_fold_out(means).max())})
    pd.DataFrame(sensitivity).to_csv(out/"leave_one_fold_out.csv", index=False)
    pd.DataFrame(paired).to_csv(out/"paired_protocol_summary.csv", index=False)
    variability = scores.groupby(["protocol","method","n","test_fold"]).macro_f1.std().groupby(
        ["protocol","method","n"]).agg(["mean","max"]).reset_index()
    variability.to_csv(out/"within_fold_seed_sd.csv", index=False)
    # New figures preserve original exports and show that the score range is near ceiling.
    fig, ax = plt.subplots(figsize=(6.8,3.6), constrained_layout=True)
    curve = score_summary[(score_summary.protocol == "validation_selected") & (score_summary.method == "clean") & (score_summary["count"] == 5)]
    for fold, g in folds[(folds.protocol == "validation_selected") & (folds.method == "clean") & (folds.n <= 16)].groupby("test_fold"):
        ax.plot(g.n, g.macro_f1, color="gray", alpha=.4, linewidth=.8)
    for n, g in scores[(scores.protocol == "validation_selected") & (scores.method == "clean") & (scores.n <= 16)].groupby("n"):
        ax.scatter(np.full(len(g),n),g.macro_f1,s=12,color="gray",alpha=.25)
    ax.errorbar(curve.n,curve["mean"],yerr=curve["std"],fmt="o-",capsize=4,label="LR: mean ± fold SD; gray = runs/fold means")
    zero_folds = zero.groupby("test_fold").apply(lambda g:manual_macro_f1(g.true_label,g.predicted_label),include_groups=False)
    ax.axhline(zero_folds.mean(),color="tab:orange",linestyle="--",label="Distinct fixed-prompt zero-shot mean")
    ax.set(xticks=[1,2,4,8,16],xlabel="Labelled original training clips per class",
           ylabel="Macro-F1 (zoomed axis)",ylim=(.81,1.015))
    ax.legend(loc="lower right",fontsize=7.5)
    figure_save(fig,out,"learning_curve_detail")
    fig,axes = plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    pairs = pd.read_csv(run/"analysis"/"paired_fold_means.csv")
    for n,g in pairs.groupby("n"):
        axes[0].scatter(np.full(len(g),n)+np.linspace(-.6,.6,len(g)),g.delta_macro_f1*100,label=f"n={n}: five fold means")
        axes[0].errorbar(n,g.delta_macro_f1.mean()*100,yerr=g.delta_macro_f1.std()*100,fmt="ks",capsize=4)
    axes[0].axhline(0,color="gray",lw=.8)
    axes[0].set(xticks=[4,16],xlabel="Original clips per class",ylabel="Noise − clean Macro-F1 (pp)",title="Paired fold differences; mean ± SD")
    axes[0].legend(fontsize=7)
    noise_summary = summaries[(summaries.protocol == "validation_selected") & (summaries.comparison == "clean_to_noise")].set_index("n")
    x = np.arange(2)
    axes[1].bar(x-.17,[noise_summary.loc[n,"repaired_instances"] for n in [4,16]],width=.34,label="Wrong → correct")
    axes[1].bar(x+.17,[noise_summary.loc[n,"harmed_instances"] for n in [4,16]],width=.34,label="Correct → wrong",color="tab:red")
    axes[1].set(xticks=x,xticklabels=["n=4","n=16"],ylabel="Prediction instances (3 seeds × 400 clips)",title="Sparse changes; repeated clips are correlated")
    axes[1].legend(fontsize=7)
    figure_save(fig,out,"augmentation_evidence")
    report = {"status":"PASS", "design":"Exploratory analysis of existing saved predictions only; no fitting/tuning",
              "independent_macro_f1_run_checks":checks,"transition_rows":len(transitions),
              "input_sha256":{str(p.relative_to(run)):sha256(p) for p in
                  [run/"predictions/zero_shot.csv",run/"predictions/few_shot.csv",run/"sensitivity_fixed_c1/predictions.csv"]},
              "source_sha256":sha256(__file__),
              "uncertainty":"Leave-one-fold-out values measure descriptive concentration, not confidence intervals; folds/seeds are correlated"}
    write_json(out/"audit.json",report)
    print(summaries[summaries.protocol == "validation_selected"].to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))
    print(pd.DataFrame(cases).to_string(index=False))


if __name__ == "__main__":
    main()
