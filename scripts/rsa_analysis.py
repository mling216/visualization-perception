"""
Representational Similarity Analysis (RSA) on the 273-image overlap.

Computes the true pairwise LPIPS perceptual distance (Zhang et al., 2018 -- a
learned distance *between two images*, not a per-image feature vector) for all
273*272/2 image pairs, then correlates that distance matrix against:

  - |human VC score_i - human VC score_j|
  - |human memorability score_i - memorability score_j|
  - |LLM VC score_i - LLM VC score_j|            (Claude, GPT)
  - |LLM memorability score_i - memorability score_j|  (Claude, GPT)

This tests whether images the LPIPS metric considers perceptually close are
also judged similarly by humans / LLMs -- i.e. whether "perceptual distance"
as a generic vision model defines it lines up with distances in each rating
space. Significance is assessed with a Mantel test (permute image labels,
not pairs, since pairwise-distance entries are not independent).

Reads:
    data/vc_270.csv, data/memorability_270.csv           (ground truth, 273 shared images)
    data/image_cache/<imageName>                          (cached images)
    results/vc/vc_<model>_scores.csv                      (LLM VC predictions)
    results/memorability270/memorability270_<model>_scores.csv

Outputs:
    results/rsa_273_overlap.json
    results/rsa_lpips_distance_matrix.npz   (image_names, distance matrix -- reusable)

Usage:
    python scripts/rsa_analysis.py --net alex
"""

import argparse
import json
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = DATA_DIR / "image_cache"

IMAGE_SIZE = 224
N_PERM = 5000
SEED = 42

PREPROCESS = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # LPIPS expects [-1, 1]
])


def cached_image_path(image_name: str) -> Path:
    return CACHE_DIR / Path(image_name).name


def load_images(image_names: list, device: torch.device) -> torch.Tensor:
    tensors = []
    for name in tqdm(image_names, desc="Loading images"):
        img = Image.open(cached_image_path(name)).convert("RGB")
        tensors.append(PREPROCESS(img))
    return torch.stack(tensors).to(device)


def pairwise_lpips_distance(images: torch.Tensor, model: lpips.LPIPS, batch_size: int = 32) -> np.ndarray:
    """Full symmetric NxN LPIPS distance matrix via the upper triangle of pairs."""
    n = images.shape[0]
    idx_i, idx_j = np.triu_indices(n, k=1)
    n_pairs = len(idx_i)
    dists = np.empty(n_pairs, dtype=np.float64)

    for start in tqdm(range(0, n_pairs, batch_size), desc="Computing pairwise LPIPS"):
        end = min(start + batch_size, n_pairs)
        bi = idx_i[start:end]
        bj = idx_j[start:end]
        with torch.no_grad():
            d = model(images[bi], images[bj])
        dists[start:end] = d.squeeze().cpu().numpy().reshape(-1)

    return squareform(dists)  # NxN, zero diagonal


def score_distance_matrix(scores: np.ndarray) -> np.ndarray:
    return np.abs(scores[:, None] - scores[None, :])


def mantel_test(dist_a: np.ndarray, dist_b: np.ndarray, n_perm: int = N_PERM, seed: int = SEED):
    """Mantel test: correlate upper-triangle entries, then assess significance by
    permuting the *labels* (rows/cols) of one matrix, preserving its internal
    dependency structure, rather than permuting pairs independently."""
    n = dist_a.shape[0]
    iu = np.triu_indices(n, k=1)
    a_flat = dist_a[iu]
    b_flat = dist_b[iu]

    r_obs, _ = pearsonr(a_flat, b_flat)
    rho_obs, _ = spearmanr(a_flat, b_flat)

    rng = np.random.default_rng(seed)
    perm_r = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(n)
        b_perm = dist_b[np.ix_(perm, perm)][iu]
        perm_r[k], _ = pearsonr(a_flat, b_perm)

    p_value = float(np.mean(np.abs(perm_r) >= np.abs(r_obs)))
    return {
        "pearson_r": float(r_obs),
        "spearman_r": float(rho_obs),
        "mantel_p": p_value,
        "n_perm": n_perm,
    }


def main():
    parser = argparse.ArgumentParser(description="RSA: pairwise LPIPS distance vs. human/LLM score distances")
    parser.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"],
                         help="Backbone for the LPIPS metric (alex is the network Zhang et al. found best matches human judgments)")
    parser.add_argument("--n-perm", type=int, default=N_PERM, help="Number of Mantel-test permutations")
    args = parser.parse_args()

    vc_df = pd.read_csv(DATA_DIR / "vc_270.csv")
    mem_df = pd.read_csv(DATA_DIR / "memorability_270.csv")
    assert set(vc_df.imageName) == set(mem_df.imageName), "VC and memorability 270 sets must share the same images"

    # Canonical image order used for every matrix below.
    image_names = vc_df["imageName"].tolist()
    vc_gt = vc_df.set_index("imageName").loc[image_names, "gt_score"].values.astype(float)
    mem_gt = mem_df.set_index("imageName").loc[image_names, "gt_score"].values.astype(float)

    def load_llm_scores(perception_dir: str, prefix: str, model_suffix: str) -> np.ndarray:
        path = RESULTS_DIR / perception_dir / f"{prefix}_{model_suffix}_scores.csv"
        df = pd.read_csv(path).set_index("imageName")["score"]
        return df.loc[image_names].values.astype(float)

    llm_vc = {
        "Claude Sonnet 4.6": load_llm_scores("vc", "vc", "claude-sonnet-4-6"),
        "GPT-5.4": load_llm_scores("vc", "vc", "gpt-5.4"),
    }
    llm_mem = {
        "Claude Sonnet 4.6": load_llm_scores("memorability270", "memorability270", "claude-sonnet-4-6"),
        "GPT-5.4": load_llm_scores("memorability270", "memorability270", "gpt-5.4"),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading {len(image_names)} images...")
    images = load_images(image_names, device)

    print(f"Building LPIPS ({args.net}) model...")
    model = lpips.LPIPS(net=args.net).to(device)

    print("Computing pairwise LPIPS distance matrix...")
    lpips_dist = pairwise_lpips_distance(images, model)

    out_npz = RESULTS_DIR / "rsa_lpips_distance_matrix.npz"
    np.savez(out_npz, image_names=np.array(image_names), distance_matrix=lpips_dist, net=args.net)
    print(f"Saved distance matrix to {out_npz}")

    results = {"net": args.net, "n_images": len(image_names), "n_perm": args.n_perm}

    print("\nRSA: LPIPS distance vs. score-difference matrices")
    for label, gt in [("Human VC", vc_gt), ("Human Memorability", mem_gt)]:
        d = score_distance_matrix(gt)
        res = mantel_test(lpips_dist, d, n_perm=args.n_perm)
        results[label] = res
        print(f"  {label}: r={res['pearson_r']:.3f}, rho={res['spearman_r']:.3f}, Mantel p={res['mantel_p']:.4f}")

    for model_name, scores in llm_vc.items():
        d = score_distance_matrix(scores)
        res = mantel_test(lpips_dist, d, n_perm=args.n_perm)
        key = f"LLM VC ({model_name})"
        results[key] = res
        print(f"  {key}: r={res['pearson_r']:.3f}, rho={res['spearman_r']:.3f}, Mantel p={res['mantel_p']:.4f}")

    for model_name, scores in llm_mem.items():
        d = score_distance_matrix(scores)
        res = mantel_test(lpips_dist, d, n_perm=args.n_perm)
        key = f"LLM Memorability ({model_name})"
        results[key] = res
        print(f"  {key}: r={res['pearson_r']:.3f}, rho={res['spearman_r']:.3f}, Mantel p={res['mantel_p']:.4f}")

    out_json = RESULTS_DIR / "rsa_273_overlap.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved RSA results to {out_json}")


if __name__ == "__main__":
    main()
