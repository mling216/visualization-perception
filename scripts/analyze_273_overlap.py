"""
Analysis for the 273-image overlap: LLM vs. VGG comparison table, bootstrap CIs,
and a check of whether VGG-predicted VC explains the LLM VC/memorability conflation.

Reads:
    data/vc_270.csv, data/memorability_270.csv                       (ground truth)
    results/vc/vc_<model>_scores.csv                                  (LLM VC predictions, superset)
    results/memorability270/memorability270_<model>_scores.csv        (LLM memorability predictions)
    results/vc270/deep_feature_baseline_vgg.csv                       (VGG VC predictions, 273)
    results/memorability270/deep_feature_baseline_vgg.csv             (VGG memorability predictions, 273)

Outputs:
    results/overlap_273_comparison.csv   (comparison table with bootstrap CIs)
    results/overlap_273_conflation.json  (VGG-predicted-VC vs LLM-memorability correlations)

Usage:
    python scripts/analyze_273_overlap.py
"""

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

LLM_MODELS = {
    "Claude Sonnet 4.6": "claude-sonnet-4-6",
    "GPT-5.4": "gpt-5.4",
}


def bootstrap_ci(x: np.ndarray, y: np.ndarray, stat_fn, n_boot: int = N_BOOT, seed: int = SEED):
    """Percentile bootstrap CI for a correlation statistic between x and y."""
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = stat_fn(x[idx], y[idx])[0]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def load_llm_overlap(perception_dir: str, prefix: str, model_suffix: str, gt_path: Path) -> pd.DataFrame:
    gt = pd.read_csv(gt_path)[["imageName", "gt_score"]]
    pred_path = RESULTS_DIR / perception_dir / f"{prefix}_{model_suffix}_scores.csv"
    pred = pd.read_csv(pred_path)[["imageName", "score"]]
    return gt.merge(pred, on="imageName", how="inner")


def load_vgg_overlap(perception_dir: str, pred_col: str, extractor: str = "vgg") -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / perception_dir / f"deep_feature_baseline_{extractor}.csv")
    return df.rename(columns={pred_col: "score"})[["imageName", "gt_score", "score"]]


def main():
    rows = []

    sources = {
        "VC": [
            ("Claude Sonnet 4.6", lambda: load_llm_overlap("vc", "vc", "claude-sonnet-4-6", DATA_DIR / "vc_270.csv")),
            ("GPT-5.4", lambda: load_llm_overlap("vc", "vc", "gpt-5.4", DATA_DIR / "vc_270.csv")),
            ("VGG19 + Ridge", lambda: load_vgg_overlap("vc270", "ridge_pred")),
            ("VGG19 + 5-NN", lambda: load_vgg_overlap("vc270", "knn_pred")),
            ("VGG19 + GradBoost", lambda: load_vgg_overlap("vc270", "gb_pred")),
            ("LPIPS + Ridge", lambda: load_vgg_overlap("vc270", "ridge_pred", "lpips")),
            ("LPIPS + 5-NN", lambda: load_vgg_overlap("vc270", "knn_pred", "lpips")),
            ("LPIPS + GradBoost", lambda: load_vgg_overlap("vc270", "gb_pred", "lpips")),
        ],
        "Memorability": [
            ("Claude Sonnet 4.6", lambda: load_llm_overlap("memorability270", "memorability270", "claude-sonnet-4-6", DATA_DIR / "memorability_270.csv")),
            ("GPT-5.4", lambda: load_llm_overlap("memorability270", "memorability270", "gpt-5.4", DATA_DIR / "memorability_270.csv")),
            ("VGG19 + Ridge", lambda: load_vgg_overlap("memorability270", "ridge_pred")),
            ("VGG19 + 5-NN", lambda: load_vgg_overlap("memorability270", "knn_pred")),
            ("VGG19 + GradBoost", lambda: load_vgg_overlap("memorability270", "gb_pred")),
            ("LPIPS + Ridge", lambda: load_vgg_overlap("memorability270", "ridge_pred", "lpips")),
            ("LPIPS + 5-NN", lambda: load_vgg_overlap("memorability270", "knn_pred", "lpips")),
            ("LPIPS + GradBoost", lambda: load_vgg_overlap("memorability270", "gb_pred", "lpips")),
        ],
    }

    vgg_vc_pred = None
    llm_mem_scores = {}

    for perception, model_list in sources.items():
        for model_name, loader in model_list:
            merged = loader().dropna(subset=["gt_score", "score"])
            x = merged["gt_score"].values.astype(float)
            y = merged["score"].values.astype(float)

            pearson_r, _ = pearsonr(x, y)
            spearman_r, _ = spearmanr(x, y)
            pearson_lo, pearson_hi = bootstrap_ci(x, y, pearsonr)
            spearman_lo, spearman_hi = bootstrap_ci(x, y, spearmanr)

            rows.append({
                "perception": perception,
                "model": model_name,
                "n": len(merged),
                "pearson_r": round(pearson_r, 3),
                "pearson_ci_lo": round(pearson_lo, 3),
                "pearson_ci_hi": round(pearson_hi, 3),
                "spearman_r": round(spearman_r, 3),
                "spearman_ci_lo": round(spearman_lo, 3),
                "spearman_ci_hi": round(spearman_hi, 3),
            })
            print(f"[{perception}] {model_name}: r={pearson_r:.3f} [{pearson_lo:.3f}, {pearson_hi:.3f}], "
                  f"rho={spearman_r:.3f} [{spearman_lo:.3f}, {spearman_hi:.3f}], n={len(merged)}")

            if perception == "VC" and model_name == "VGG19 + Ridge":
                vgg_vc_pred = merged[["imageName", "score"]].rename(columns={"score": "vgg_vc_pred"})
            if perception == "Memorability" and model_name in LLM_MODELS:
                llm_mem_scores[model_name] = merged[["imageName", "score"]].rename(columns={"score": f"llm_mem_{model_name}"})

    out_csv = RESULTS_DIR / "overlap_273_comparison.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved comparison table to {out_csv}")

    # VC/memorability conflation check: does VGG-predicted VC correlate with LLM memorability?
    conflation = {}
    for model_name, mem_df in llm_mem_scores.items():
        merged = vgg_vc_pred.merge(mem_df, on="imageName", how="inner")
        x = merged["vgg_vc_pred"].values.astype(float)
        y = merged[f"llm_mem_{model_name}"].values.astype(float)
        r, _ = pearsonr(x, y)
        rho, _ = spearmanr(x, y)
        r_lo, r_hi = bootstrap_ci(x, y, pearsonr)
        conflation[model_name] = {
            "pearson_r": round(r, 3),
            "pearson_ci": [round(r_lo, 3), round(r_hi, 3)],
            "spearman_r": round(rho, 3),
            "n": len(merged),
        }
        print(f"\nVGG-predicted VC vs {model_name} memorability: r={r:.3f} [{r_lo:.3f}, {r_hi:.3f}], rho={rho:.3f}, n={len(merged)}")

    out_json = RESULTS_DIR / "overlap_273_conflation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(conflation, f, indent=2)
    print(f"Saved conflation check to {out_json}")


if __name__ == "__main__":
    main()
