"""Analyze three repeated LLM runs on the 273-image VC/memorability overlap."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
N_BOOT = 5000
SEED = 42

MODELS = {
    "gpt-5.4": "GPT-5.4",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
}

PERCEPTIONS = {
    "vc": {
        "label": "VC",
        "data": DATA_DIR / "vc_270.csv",
        "result_dir": RESULTS_DIR / "vc",
        "run_prefix": "vc",
    },
    "memorability": {
        "label": "Memorability",
        "data": DATA_DIR / "memorability_270.csv",
        "result_dir": RESULTS_DIR / "memorability",
        "run_prefix": "memorability",
    },
}

BASELINES = {
    "vc": [
        ("VGG19 + Ridge", RESULTS_DIR / "vc270" / "deep_feature_baseline_vgg.csv"),
        ("LPIPS + Ridge", RESULTS_DIR / "vc270" / "deep_feature_baseline_lpips.csv"),
    ],
    "memorability": [
        ("VGG19 + Ridge", RESULTS_DIR / "memorability270" / "deep_feature_baseline_vgg.csv"),
        ("LPIPS + Ridge", RESULTS_DIR / "memorability270" / "deep_feature_baseline_lpips.csv"),
    ],
}


def correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if kind == "pearson":
        return float(pearsonr(x, y)[0])
    return float(spearmanr(x, y)[0])


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    kind: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = len(x)
    values = np.empty(N_BOOT)
    for index in range(N_BOOT):
        sample = rng.integers(0, n, n)
        values[index] = correlation(x[sample], y[sample], kind)
    return tuple(np.percentile(values, [2.5, 97.5]))


def run_paths(perception: str, model_tag: str) -> list[Path]:
    spec = PERCEPTIONS[perception]
    run1_dir = spec["result_dir"]
    run1_prefix = spec["run_prefix"]
    if perception == "memorability":
        run1_dir = RESULTS_DIR / "memorability270"
        run1_prefix = "memorability270"
    return [
        run1_dir / f"{run1_prefix}_{model_tag}_scores.csv",
        spec["result_dir"] / "runs" / f"{spec['run_prefix']}_{model_tag}_run2_scores.csv",
        spec["result_dir"] / "runs" / f"{spec['run_prefix']}_{model_tag}_run3_scores.csv",
    ]


def load_runs(perception: str, model_tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = PERCEPTIONS[perception]
    ground_truth = pd.read_csv(spec["data"])[["imageName", "gt_score"]]
    frames = []
    for run_id, path in enumerate(run_paths(perception, model_tag), 1):
        if not path.exists():
            raise FileNotFoundError(f"Missing run {run_id}: {path}")
        frame = pd.read_csv(path)[["imageName", "score"]].copy()
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        frame = frame.rename(columns={"score": f"run_{run_id}"})
        frames.append(frame)
    runs = frames[0]
    for frame in frames[1:]:
        runs = runs.merge(frame, on="imageName", how="inner", validate="one_to_one")
    runs = ground_truth.merge(runs, on="imageName", how="inner", validate="one_to_one")
    if len(runs) != 273:
        raise ValueError(f"{perception}/{model_tag} has {len(runs)} complete images, expected 273")
    runs["mean_score"] = runs[["run_1", "run_2", "run_3"]].mean(axis=1)
    runs["run_sd"] = runs[["run_1", "run_2", "run_3"]].std(axis=1, ddof=1)
    return ground_truth, runs


def summarize_model(
    perception: str,
    model_tag: str,
    model_label: str,
    runs: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[list[dict], dict]:
    rows = []
    for run_id in [1, 2, 3]:
        prediction = runs[f"run_{run_id}"].to_numpy(float)
        human = runs["gt_score"].to_numpy(float)
        rows.extend([
            {
                "perception": perception,
                "model": model_label,
                "estimate": f"run_{run_id}",
                "run_id": run_id,
                "n": len(runs),
                "pearson_r": correlation(human, prediction, "pearson"),
                "spearman_rho": correlation(human, prediction, "spearman"),
            }
        ])

    human = runs["gt_score"].to_numpy(float)
    mean_prediction = runs["mean_score"].to_numpy(float)
    mean_row = {
        "perception": perception,
        "model": model_label,
        "estimate": "three_run_mean",
        "run_id": "mean",
        "n": len(runs),
        "pearson_r": correlation(human, mean_prediction, "pearson"),
        "spearman_rho": correlation(human, mean_prediction, "spearman"),
    }
    for kind, key in [("pearson", "pearson"), ("spearman", "spearman")]:
        low, high = bootstrap_ci(human, mean_prediction, kind, rng)
        mean_row[f"{key}_ci_low"] = low
        mean_row[f"{key}_ci_high"] = high
    rows.append(mean_row)

    run_values = runs[["run_1", "run_2", "run_3"]].to_numpy(float)
    reliability_rows = []
    for first in range(3):
        for second in range(first + 1, 3):
            reliability_rows.append({
                "perception": perception,
                "model": model_label,
                "run_a": first + 1,
                "run_b": second + 1,
                "pearson_r": correlation(run_values[:, first], run_values[:, second], "pearson"),
                "spearman_rho": correlation(run_values[:, first], run_values[:, second], "spearman"),
            })
    reliability = pd.DataFrame(reliability_rows)
    reliability_summary = {
        "perception": perception,
        "model": model_label,
        "mean_pairwise_pearson": float(reliability["pearson_r"].mean()),
        "mean_pairwise_spearman": float(reliability["spearman_rho"].mean()),
        "median_image_run_sd": float(runs["run_sd"].median()),
        "mean_image_run_sd": float(runs["run_sd"].mean()),
        "p90_image_run_sd": float(runs["run_sd"].quantile(0.9)),
        "max_image_run_sd": float(runs["run_sd"].max()),
        "mean_score": float(runs["mean_score"].mean()),
        "unique_mean_scores": int(runs["mean_score"].nunique()),
        "largest_mean_score_band": float(runs["mean_score"].value_counts(normalize=True).iloc[0]),
    }
    banding = []
    for estimate in ["run_1", "run_2", "run_3", "mean_score"]:
        values = runs[estimate]
        banding.append({
            "perception": perception,
            "model": model_label,
            "estimate": estimate,
            "unique_scores": int(values.nunique()),
            "largest_score_band_fraction": float(values.value_counts(normalize=True).iloc[0]),
        })
    return rows, {"pairwise": reliability, "summary": reliability_summary, "banding": banding}


def compare_models(
    perception: str,
    gpt_runs: pd.DataFrame,
    claude_runs: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    human = gpt_runs["gt_score"].to_numpy(float)
    gpt = gpt_runs["mean_score"].to_numpy(float)
    claude = claude_runs["mean_score"].to_numpy(float)
    output = {"perception": perception}
    for kind in ["pearson", "spearman"]:
        observed = correlation(human, gpt, kind) - correlation(human, claude, kind)
        values = np.empty(N_BOOT)
        for index in range(N_BOOT):
            sample = rng.integers(0, len(human), len(human))
            values[index] = correlation(human[sample], gpt[sample], kind) - correlation(human[sample], claude[sample], kind)
        low, high = np.percentile(values, [2.5, 97.5])
        output[f"gpt_minus_claude_{kind}"] = float(observed)
        output[f"gpt_minus_claude_{kind}_ci_low"] = float(low)
        output[f"gpt_minus_claude_{kind}_ci_high"] = float(high)
    return output


def compare_to_baseline(
    perception: str,
    model_label: str,
    llm_runs: pd.DataFrame,
    baseline_label: str,
    baseline_path: Path,
    rng: np.random.Generator,
) -> dict:
    baseline = pd.read_csv(baseline_path)[["imageName", "ridge_pred"]]
    merged = llm_runs.merge(baseline, on="imageName", how="inner", validate="one_to_one")
    human = merged["gt_score"].to_numpy(float)
    llm = merged["mean_score"].to_numpy(float)
    baseline_scores = merged["ridge_pred"].to_numpy(float)
    output = {"perception": perception, "llm": model_label, "baseline": baseline_label}
    for kind in ["pearson", "spearman"]:
        observed = correlation(human, llm, kind) - correlation(human, baseline_scores, kind)
        values = np.empty(N_BOOT)
        for index in range(N_BOOT):
            sample = rng.integers(0, len(human), len(human))
            values[index] = correlation(human[sample], llm[sample], kind) - correlation(human[sample], baseline_scores[sample], kind)
        low, high = np.percentile(values, [2.5, 97.5])
        output[f"llm_minus_baseline_{kind}"] = float(observed)
        output[f"llm_minus_baseline_{kind}_ci_low"] = float(low)
        output[f"llm_minus_baseline_{kind}_ci_high"] = float(high)
    return output


def type_stratified(perception: str, model_runs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    data_path = PERCEPTIONS[perception]["data"]
    target = pd.read_csv(data_path)[["imageName", "gt_chart_type"]]
    rows = []
    for model_tag, model_label in MODELS.items():
        frame = target.merge(model_runs[(perception, model_tag)], on="imageName", how="inner")
        chart_types = frame["gt_chart_type"].astype(str).str.lower()
        is_bar_line = chart_types.str.contains("bar") & chart_types.str.contains("line")
        frame = frame.loc[~is_bar_line].copy()
        frame["group"] = frame["gt_chart_type"].astype(str).str.split(";").str[0].str.strip()
        for group, subset in frame.groupby("group"):
            if len(subset) < 12:
                continue
            rows.append({
                "perception": perception,
                "model": model_label,
                "group": group,
                "n": len(subset),
                "spearman_rho": correlation(
                    subset["gt_score"].to_numpy(float),
                    subset["mean_score"].to_numpy(float),
                    "spearman",
                ),
            })
    return pd.DataFrame(rows)


def cross_construct_correlations(model_runs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    vc = model_runs[("vc", "gpt-5.4")][["imageName", "gt_score"]].rename(columns={"gt_score": "human_vc"})
    memorability = model_runs[("memorability", "gpt-5.4")][["imageName", "gt_score"]].rename(columns={"gt_score": "human_memorability"})
    merged = vc.merge(memorability, on="imageName", how="inner", validate="one_to_one")
    rows = []
    observers = {
        "Human": (merged["human_vc"], merged["human_memorability"]),
    }
    for model_tag, model_label in MODELS.items():
        vc_predictions = model_runs[("vc", model_tag)][["imageName", "mean_score"]].rename(columns={"mean_score": "vc_prediction"})
        memorability_predictions = model_runs[("memorability", model_tag)][["imageName", "mean_score"]].rename(columns={"mean_score": "memorability_prediction"})
        predictions = merged[["imageName"]].merge(vc_predictions, on="imageName", how="inner", validate="one_to_one")
        predictions = predictions.merge(memorability_predictions, on="imageName", how="inner", validate="one_to_one")
        observers[model_label] = (predictions["vc_prediction"], predictions["memorability_prediction"])
    for observer, (vc_values, memorability_values) in observers.items():
        rows.append({
            "observer": observer,
            "n": len(vc_values),
            "pearson_r": correlation(vc_values.to_numpy(float), memorability_values.to_numpy(float), "pearson"),
            "spearman_rho": correlation(vc_values.to_numpy(float), memorability_values.to_numpy(float), "spearman"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    correlation_rows = []
    reliability_rows = []
    reliability_summary = []
    banding_rows = []
    model_runs = {}
    baseline_comparisons = []
    type_rows = []

    for perception in PERCEPTIONS:
        for model_tag, model_label in MODELS.items():
            _, runs = load_runs(perception, model_tag)
            model_runs[(perception, model_tag)] = runs
            rows, reliability = summarize_model(perception, model_tag, model_label, runs, rng)
            correlation_rows.extend(rows)
            reliability_rows.extend(reliability["pairwise"].to_dict("records"))
            reliability_summary.append(reliability["summary"])
            banding_rows.extend(reliability["banding"])

        for model_tag, model_label in MODELS.items():
            for baseline_label, baseline_path in BASELINES[perception]:
                baseline_comparisons.append(compare_to_baseline(
                    perception,
                    model_label,
                    model_runs[(perception, model_tag)],
                    baseline_label,
                    baseline_path,
                    rng,
                ))
        type_rows.extend(type_stratified(perception, model_runs).to_dict("records"))

    comparisons = []
    for perception in PERCEPTIONS:
        comparisons.append(compare_models(
            perception,
            model_runs[(perception, "gpt-5.4")],
            model_runs[(perception, "claude-sonnet-4-6")],
            rng,
        ))

    cross_construct = cross_construct_correlations(model_runs)

    output_dir = RESULTS_DIR / "three_run_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(correlation_rows).to_csv(output_dir / "correlations.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(output_dir / "run_pairwise_reliability.csv", index=False)
    pd.DataFrame(reliability_summary).to_csv(output_dir / "reliability_summary.csv", index=False)
    pd.DataFrame(banding_rows).to_csv(output_dir / "score_banding.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output_dir / "gpt_minus_claude_bootstrap.csv", index=False)
    pd.DataFrame(baseline_comparisons).to_csv(output_dir / "llm_minus_baseline_bootstrap.csv", index=False)
    pd.DataFrame(type_rows).to_csv(output_dir / "type_stratified_mean_scores.csv", index=False)
    cross_construct.to_csv(output_dir / "cross_construct_correlations.csv", index=False)

    report = {
        "n_images": 273,
        "n_runs": 3,
        "bootstrap_replicates": N_BOOT,
        "seed": SEED,
        "note": "Bootstrap intervals resample images; run-to-run variation is summarized separately.",
        "correlations": correlation_rows,
        "reliability": reliability_summary,
        "model_comparisons": comparisons,
        "baseline_comparisons": baseline_comparisons,
        "cross_construct_correlations": cross_construct.to_dict("records"),
    }
    with open(output_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Wrote multi-round analysis to {output_dir}")
    print(pd.DataFrame(correlation_rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nReliability summary")
    print(pd.DataFrame(reliability_summary).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nGPT minus Claude bootstrap comparisons")
    print(pd.DataFrame(comparisons).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()