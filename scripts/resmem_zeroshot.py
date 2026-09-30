"""Zero-shot ResMem baseline for memorability on the 273 shared images.

ResMem (Needell & Bainbridge) is a memorability model trained on natural
photographs. We apply it to the visualization images as-is, with no fitting,
to see whether a memorability-specific model transfers across image domains.

Reads data/memorability_270.csv and the cached images in data/image_cache/.
Writes results/memorability270/resmem_zeroshot.csv (per-image predictions)
and results/resmem_zeroshot_metrics.json (Pearson/Spearman with 95% bootstrap
intervals, 5,000 resamples, seed 42).

Usage:
    python scripts/resmem_zeroshot.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from resmem import ResMem, transformer
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = DATA_DIR / "image_cache"

N_BOOT = 5000
SEED = 42


def cached_image_path(image_name: str) -> Path:
    return CACHE_DIR / Path(image_name).name


def bootstrap_ci(x: np.ndarray, y: np.ndarray, stat_fn, n_boot: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = stat_fn(x[idx], y[idx])[0]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def run_resmem(df: pd.DataFrame, model: ResMem, device: torch.device) -> pd.DataFrame:
    preds = []
    for name in tqdm(df["imageName"], desc="ResMem inference"):
        img = Image.open(cached_image_path(name)).convert("RGB")
        x = transformer(img).view(-1, 3, 227, 227).to(device)
        with torch.no_grad():
            pred = model(x).item()
        preds.append(pred)
    out = df[["imageName", "gt_score"]].copy()
    out["resmem_score"] = preds
    return out


def evaluate(out: pd.DataFrame, label: str) -> dict:
    x = out["gt_score"].values.astype(float)
    y = out["resmem_score"].values.astype(float)
    pearson_r, _ = pearsonr(x, y)
    spearman_r, _ = spearmanr(x, y)
    pearson_lo, pearson_hi = bootstrap_ci(x, y, pearsonr)
    spearman_lo, spearman_hi = bootstrap_ci(x, y, spearmanr)
    print(f"[{label}] n={len(out)} r={pearson_r:.3f} [{pearson_lo:.3f}, {pearson_hi:.3f}] "
          f"rho={spearman_r:.3f} [{spearman_lo:.3f}, {spearman_hi:.3f}]")
    return {
        "n": len(out),
        "pearson_r": round(float(pearson_r), 3),
        "pearson_ci": [round(pearson_lo, 3), round(pearson_hi, 3)],
        "spearman_r": round(float(spearman_r), 3),
        "spearman_ci": [round(spearman_lo, 3), round(spearman_hi, 3)],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResMem(pretrained=True).to(device)
    model.eval()

    metrics = {}
    df_270 = pd.read_csv(DATA_DIR / "memorability_270.csv").dropna(subset=["gt_score"])
    out_270 = run_resmem(df_270, model, device)
    out_dir_270 = RESULTS_DIR / "memorability270"
    out_dir_270.mkdir(parents=True, exist_ok=True)
    out_270.to_csv(out_dir_270 / "resmem_zeroshot.csv", index=False)
    metrics["memorability_270"] = evaluate(out_270, "memorability_270")

    out_json = RESULTS_DIR / "resmem_zeroshot_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {out_json}")


if __name__ == "__main__":
    main()
