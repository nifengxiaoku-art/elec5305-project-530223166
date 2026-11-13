"""Readable single-panel count/recall plot from one prediction per original clip."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .dataset import CATEGORIES, sha256
from .evaluate import metrics
from .analyze_study import figure_save
from .extract_embeddings import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    for extension in [".png", ".pdf", ".json"]:
        if args.output_prefix.with_suffix(extension).exists():
            raise FileExistsError("Choose new plot paths; preserve existing results")
    data = pd.read_csv(args.predictions)
    if len(data) != 400 or data.filename.nunique() != 400:
        raise ValueError("Expected 400 unique original predictions, not repeated seeds")
    result = metrics(data.true_label, data.predicted_label)
    counts = np.asarray(result["confusion_matrix"])
    if not np.all(counts.sum(axis=1) == 40):
        raise ValueError("Expected 40 original test clips in every class")
    recall = counts/counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4,4.9), constrained_layout=True)
    image = ax.imshow(recall, vmin=0, vmax=1, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if counts[i,j]:
                ax.text(j,i,f"{counts[i,j]}\n({recall[i,j]*100:.1f}%)",
                        ha="center",va="center",fontsize=7,
                        color="white" if recall[i,j] > .6 else "black")
    ax.set(xticks=range(10),yticks=range(10),xticklabels=CATEGORIES,yticklabels=CATEGORIES,
           xlabel="Predicted category",ylabel="True category",
           title="Fixed-prompt CLAP: count and within-row percentage")
    ax.tick_params(axis="x",rotation=50,labelsize=8)
    ax.tick_params(axis="y",labelsize=8)
    fig.colorbar(image, ax=ax, label="Fraction of true-class clips", shrink=.82)
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    figure_save(fig,args.output_prefix.parent,args.output_prefix.name)
    write_json(args.output_prefix.with_suffix(".json"),
               {"input_sha256":sha256(args.predictions),"source_sha256":sha256(__file__),
                "originals":400,"annotations":"counts and percentages within each true-class row",**result})
    print("Readable count/recall confusion figure created from 400 unique originals")


if __name__ == "__main__":
    main()
