"""Observer-convergence analysis: LLM-baseline agreement vs. LLM-human agreement.

Reports paired image-bootstrap intervals for the dependent-correlation difference
r(LLM, baseline) - r(LLM, human) on the shared 273-image set.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
N_BOOT = 5000
SEED = 42

MODELS = {"gpt-5.4": "GPT-5.4", "claude-sonnet-4-6": "Claude Sonnet 4.6"}
SETS = {
    "vc": ("VC", ROOT / "data" / "vc_270.csv", ROOT / "results" / "vc270"),
    "memorability": ("Memorability", ROOT / "data" / "memorability_270.csv",
                     ROOT / "results" / "memorability270"),
}
BASELINES = {"LPIPS-style + Ridge": "deep_feature_baseline_lpips.csv",
             "VGG19 + Ridge": "deep_feature_baseline_vgg.csv"}


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation for stacked bootstrap samples."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for key, (label, data_path, baseline_dir) in SETS.items():
        human = pd.read_csv(data_path)[["imageName", "gt_score"]]
        for tag, model in MODELS.items():
            llm = pd.read_csv(
                ROOT / "results" / key / "aggregated" / f"{key}_{tag}_3run_mean_scores.csv"
            )[["imageName", "score"]]
            for baseline_label, filename in BASELINES.items():
                baseline = pd.read_csv(baseline_dir / filename)[["imageName", "ridge_pred"]]
                merged = human.merge(llm, on="imageName").merge(baseline, on="imageName")
                h = merged.gt_score.to_numpy(float)
                m = merged.score.to_numpy(float)
                b = merged.ridge_pred.to_numpy(float)
                n = len(merged)

                idx = rng.integers(0, n, (N_BOOT, n))
                diff = pearson_rows(m[idx], b[idx]) - pearson_rows(m[idx], h[idx])
                low, high = np.percentile(diff, [2.5, 97.5])
                rows.append({
                    "construct": label,
                    "model": model,
                    "baseline": baseline_label,
                    "n": n,
                    "r_llm_human": pearson_rows(m[None], h[None])[0],
                    "r_baseline_human": pearson_rows(b[None], h[None])[0],
                    "r_llm_baseline": pearson_rows(m[None], b[None])[0],
                    "diff_llm_baseline_minus_llm_human": pearson_rows(m[None], b[None])[0]
                    - pearson_rows(m[None], h[None])[0],
                    "diff_ci_low": low,
                    "diff_ci_high": high,
                })

    frame = pd.DataFrame(rows)
    out_dir = ROOT / "results" / "three_run_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "observer_convergence.csv", index=False)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
