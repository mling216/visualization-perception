"""Bootstrap observer convergence against human alignment."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).parent.parent
N_BOOT = 5000
SEED = 42
MODELS = {
    "gpt-5.4": "GPT-5.4",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
}
SETS = {
    "vc": (ROOT / "data" / "vc_270.csv", ROOT / "results" / "vc270" / "deep_feature_baseline_lpips.csv"),
    "memorability": (
        ROOT / "data" / "memorability_270.csv",
        ROOT / "results" / "memorability270" / "deep_feature_baseline_lpips.csv",
    ),
}


def pearson_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    return (left * right).sum(axis=1) / np.sqrt((left * left).sum(axis=1) * (right * right).sum(axis=1))


def spearman_bootstrap(left: np.ndarray, right: np.ndarray, samples: np.ndarray) -> np.ndarray:
    values = np.empty(len(samples))
    for index, sample in enumerate(samples):
        values[index] = np.corrcoef(rankdata(left[sample]), rankdata(right[sample]))[0, 1]
    return values


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for perception, (data_path, baseline_path) in SETS.items():
        human = pd.read_csv(data_path)[["imageName", "gt_score"]]
        baseline = pd.read_csv(baseline_path)[["imageName", "ridge_pred"]]
        for model_tag, model_label in MODELS.items():
            prediction_path = ROOT / "results" / perception / "aggregated" / f"{perception}_{model_tag}_3run_mean_scores.csv"
            prediction = pd.read_csv(prediction_path)[["imageName", "score"]]
            frame = human.merge(prediction, on="imageName", validate="one_to_one").merge(
                baseline, on="imageName", validate="one_to_one"
            )
            human_values = frame["gt_score"].to_numpy(float)
            llm_values = frame["score"].to_numpy(float)
            baseline_values = frame["ridge_pred"].to_numpy(float)
            samples = rng.integers(0, len(frame), size=(N_BOOT, len(frame)))
            for metric, correlation in [("pearson", pearson_rows), ("spearman", spearman_bootstrap)]:
                if metric == "pearson":
                    llm_human_boot = correlation(llm_values[samples], human_values[samples])
                    llm_baseline_boot = correlation(llm_values[samples], baseline_values[samples])
                else:
                    llm_human_boot = correlation(llm_values, human_values, samples)
                    llm_baseline_boot = correlation(llm_values, baseline_values, samples)
                difference = llm_baseline_boot - llm_human_boot
                observed_llm_human = float(np.corrcoef(
                    rankdata(llm_values) if metric == "spearman" else llm_values,
                    rankdata(human_values) if metric == "spearman" else human_values,
                )[0, 1])
                observed_llm_baseline = float(np.corrcoef(
                    rankdata(llm_values) if metric == "spearman" else llm_values,
                    rankdata(baseline_values) if metric == "spearman" else baseline_values,
                )[0, 1])
                rows.append({
                    "perception": perception,
                    "model": model_label,
                    "metric": metric,
                    "n": len(frame),
                    "llm_human": observed_llm_human,
                    "llm_lpips": observed_llm_baseline,
                    "lpips_minus_human": observed_llm_baseline - observed_llm_human,
                    "ci_low": float(np.percentile(difference, 2.5)),
                    "ci_high": float(np.percentile(difference, 97.5)),
                })

    output = ROOT / "results" / "three_run_analysis" / "observer_convergence_bootstrap.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()