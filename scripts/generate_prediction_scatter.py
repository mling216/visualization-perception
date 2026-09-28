"""Generate the prediction scatter plot from three-run LLM mean predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr


ROOT = Path(__file__).parent.parent
FIGURES = ROOT / "figures"
ANALYSIS = ROOT / "results" / "three_run_analysis"

PANELS = [
    ("Visual complexity", "GPT-5.4", "vc", "gpt-5.4", "vc_270.csv", True),
    ("Visual complexity", "Claude Sonnet 4.6", "vc", "claude-sonnet-4-6", "vc_270.csv", True),
    ("Memorability", "GPT-5.4", "memorability", "gpt-5.4", "memorability_270.csv", True),
    ("Memorability", "Claude Sonnet 4.6", "memorability", "claude-sonnet-4-6", "memorability_270.csv", True),
    ("Aesthetic pleasure", "GPT-5.4", "beauvis", "gpt-5.4", "beauvis_15.csv", False),
    ("Aesthetic pleasure", "Claude Sonnet 4.6", "beauvis", "claude-sonnet-4-6", "beauvis_15.csv", False),
    ("Readability", "GPT-5.4", "previs", "gpt-5.4", "previs_all_9.csv", False),
    ("Readability", "Claude Sonnet 4.6", "previs", "claude-sonnet-4-6", "previs_all_9.csv", False),
]


def load_panel(perception: str, model_tag: str, data_name: str) -> pd.DataFrame:
    if perception in {"vc", "memorability"}:
        path = ROOT / "results" / perception / "aggregated" / f"{perception}_{model_tag}_3run_mean_scores.csv"
    else:
        path = ROOT / "results" / perception / f"{perception}_{model_tag}_scores.csv"
    target = pd.read_csv(ROOT / "data" / data_name)
    frame = pd.read_csv(path)
    return target[["imageName", "gt_score"]].merge(frame[["imageName", "score"]], on="imageName", validate="one_to_one")


def main() -> None:
    correlation = pd.read_csv(ANALYSIS / "correlations.csv")
    mean_rows = correlation[correlation["estimate"] == "three_run_mean"].set_index(["perception", "model"])

    fig, axes = plt.subplots(4, 2, figsize=(7.0, 11.5), sharex=False, sharey=True)
    for axis, (construct, model, perception, model_tag, data_name, has_bootstrap) in zip(axes.ravel(), PANELS):
        frame = load_panel(perception, model_tag, data_name)
        x = frame["gt_score"].to_numpy(float)
        y = frame["score"].to_numpy(float)
        axis.scatter(x, y, s=22, alpha=0.58, color="#315b78", edgecolors="white", linewidth=0.3)
        fit = linregress(x, y)
        if perception == "memorability":
            x_limits = (float(x.min()), float(x.max()))
            padding = 0.03 * (x_limits[1] - x_limits[0])
            x_limits = (x_limits[0] - padding, x_limits[1] + padding)
        else:
            x_limits = (0.0, 1.0)
        grid = np.linspace(*x_limits, 100)
        axis.plot(grid, fit.intercept + fit.slope * grid, color="#b34e3f", linewidth=1.6)
        if perception != "memorability":
            axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0)

        if has_bootstrap:
            stats = mean_rows.loc[(perception, model)]
            annotation = (
                f"r={stats.pearson_r:.2f} [{stats.pearson_ci_low:.2f}, {stats.pearson_ci_high:.2f}]\n"
                f"$\\rho$={stats.spearman_rho:.2f} [{stats.spearman_ci_low:.2f}, {stats.spearman_ci_high:.2f}]\n"
                f"n={len(frame)}"
            )
        else:
            annotation = (
                f"r={pearsonr(x, y)[0]:.2f}\n"
                f"$\\rho$={spearmanr(x, y)[0]:.2f}\n"
                f"n={len(frame)}; point estimate"
            )
        if perception == "memorability":
            annotation_position = (0.96, 0.04)
            annotation_alignment = {"va": "bottom", "ha": "right"}
        elif has_bootstrap:
            annotation_position = (0.04, 0.96)
            annotation_alignment = {"va": "top", "ha": "left"}
        else:
            annotation_position = (0.96, 0.04)
            annotation_alignment = {"va": "bottom", "ha": "right"}
        axis.text(*annotation_position, annotation, transform=axis.transAxes, fontsize=9.5,
                  **annotation_alignment,
                  bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.85, "pad": 2.5})
        axis.set_title(f"{construct} / {model}", fontsize=10.2, pad=3)
        axis.set_xlim(*x_limits)
        axis.set_ylim(0, 1)
        axis.tick_params(labelsize=9.5)
        axis.grid(True, color="0.9", linewidth=0.7)

    for axis in axes[1, :]:
        axis.set_xlabel("Human memorability (original scale)", fontsize=10.5, labelpad=4)
    for axis in axes[-1, :]:
        axis.set_xlabel("Normalized human score", fontsize=10.5, labelpad=4)
    for row, axis in enumerate(axes[:, 0]):
        axis.set_ylabel("Three-run mean LLM score" if row < 2 else "LLM score", fontsize=10.5, labelpad=5)
    fig.tight_layout(pad=1.0)
    row_shifts = {1: 0.015, 2: 0.015, 3: 0.03}
    for row, shift in row_shifts.items():
        for axis in axes[row, :]:
            position = axis.get_position()
            axis.set_position([position.x0, position.y0 + shift, position.width, position.height])
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "prediction_scatter_grid.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "prediction_scatter_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIGURES / 'prediction_scatter_grid.pdf'}")


if __name__ == "__main__":
    main()