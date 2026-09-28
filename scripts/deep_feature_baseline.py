"""
Deep-feature baseline for perceptual-attribute prediction.

Extracts frozen CNN embeddings from visualization images and predicts human
perceptual scores (memorability, visual complexity, etc.) using Ridge regression
and k-NN.  Intended as a non-LLM baseline that can be compared with the
zero-shot LLM scores produced by score_perception.py.

Usage:
    python scripts/deep_feature_baseline.py --perception memorability
    python scripts/deep_feature_baseline.py --perception vc --extractor clip
    python scripts/deep_feature_baseline.py --perception vc --extractor lpips
    python scripts/deep_feature_baseline.py --perception memorability --kfold 5 --seed 42

Outputs:
    results/<perception>/deep_feature_baseline_<extractor>.csv
    results/<perception>/deep_feature_metrics_<extractor>.json
    results/<perception>/embeddings_<extractor>.npz
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from torchvision.models import vgg19, VGG19_Weights
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = REPO_ROOT / "config"
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "data" / "image_cache"

IMAGE_SIZE = 224
BATCH_SIZE = 16


def load_config(perception: str) -> dict:
    path = CONFIG_DIR / f"{perception}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / cfg["data_csv"])
    df = df.dropna(subset=[cfg["image_col"], cfg["url_col"], cfg["gt_col"]])
    return df


def cached_image_path(image_name: str) -> Path:
    safe = Path(image_name).name
    return CACHE_DIR / safe


def fetch_image(url: str, save_path: Path, timeout: int = 60) -> bool:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
        with open(save_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"  Failed to fetch {url}: {e}")
        return False


def get_or_fetch_image(image_name: str, url: str) -> Path | None:
    cache = cached_image_path(image_name)
    if cache.exists():
        return cache
    if fetch_image(url, cache):
        return cache
    return None


# ── Extractors ───────────────────────────────────────────────────────────────

class VGGExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        weights = VGG19_Weights.DEFAULT
        self.preprocess = weights.transforms()
        # VGG19 up to relu5_1 (features index 30), then global average pool.
        self.model = vgg19(weights=weights).features[:30].eval().to(device)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def __call__(self, image_paths: list[Path]) -> np.ndarray:
        embeddings = []
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i : i + BATCH_SIZE]
            tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensors.append(self.preprocess(img))
            x = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                features = self.model(x)            # [B, 512, 14, 14]
                pooled = self.pool(features).squeeze(-1).squeeze(-1)  # [B, 512]
            embeddings.append(pooled.cpu().numpy())
        return np.vstack(embeddings)


class LPIPSExtractor:
    """LPIPS-style multi-layer VGG features: channel-normalized activations
    from early, mid, and late conv layers, spatially averaged and concatenated.

    The original LPIPS metric (Zhang et al., 2018) computes a *distance*
    between two images from VGG16 relu1_2/relu2_2/relu3_3/relu4_3/relu5_3
    activations, L2-normalized per spatial location across channels. Here we
    adapt the same normalize-then-pool idea to VGG19 (the extractor already
    used elsewhere in this script) to produce a single per-image *feature
    vector* usable by Ridge/k-NN/GB, rather than a pairwise distance.
    """

    # (layer index in vgg19().features, human-readable name)
    LAYERS = [(3, "relu1_2"), (8, "relu2_2"), (17, "relu3_4"), (26, "relu4_4"), (35, "relu5_4")]

    def __init__(self, device: torch.device):
        self.device = device
        weights = VGG19_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.model = vgg19(weights=weights).features.eval().to(device)
        self.layer_indices = {idx for idx, _ in self.LAYERS}
        self.max_layer = max(self.layer_indices)

    def _layer_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        activations = []
        h = x
        for i, layer in enumerate(self.model):
            h = layer(h)
            if i in self.layer_indices:
                activations.append(h)
            if i == self.max_layer:
                break
        return activations

    def __call__(self, image_paths: list[Path]) -> np.ndarray:
        embeddings = []
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i : i + BATCH_SIZE]
            tensors = [self.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            x = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                activations = self._layer_activations(x)
                pooled_layers = []
                for act in activations:
                    # LPIPS-style unit-normalize across the channel dimension per pixel,
                    # then average-pool spatially to get one vector per layer.
                    norm = act.norm(dim=1, keepdim=True).clamp_min(1e-10)
                    act_normed = act / norm
                    pooled = act_normed.mean(dim=(2, 3))  # [B, C]
                    pooled_layers.append(pooled)
                feat = torch.cat(pooled_layers, dim=1)  # [B, sum(C)]
            embeddings.append(feat.cpu().numpy())
        return np.vstack(embeddings)


class CLIExtractor:
    def __init__(self, device: torch.device, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.device = device
        try:
            import open_clip
        except ImportError as e:
            raise RuntimeError("open_clip is not installed; install with: pip install open-clip-torch") from e

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()

    def __call__(self, image_paths: list[Path]) -> np.ndarray:
        embeddings = []
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i : i + BATCH_SIZE]
            tensors = [self.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            x = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_image(x)
            embeddings.append(emb.cpu().numpy())
        return np.vstack(embeddings)


def build_extractor(name: str, device: torch.device):
    name = name.lower()
    if name == "vgg":
        return VGGExtractor(device)
    if name == "lpips":
        return LPIPSExtractor(device)
    if name in {"clip", "open_clip"}:
        return CLIExtractor(device)
    raise ValueError(f"Unknown extractor: {name}")


# ── Modeling ─────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)
    return {
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "mse": float(mse),
        "r2": float(r2),
        "n": int(len(y_true)),
    }


def cross_val_predict_ridge(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    alphas = np.logspace(1, 5, 20)
    for train_idx, test_idx in kf.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
        model.fit(X_train, y[train_idx])
        preds[test_idx] = model.predict(X_test)
    return preds


def cross_val_predict_knn(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int, k: int = 5) -> np.ndarray:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for train_idx, test_idx in kf.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = KNeighborsRegressor(n_neighbors=k, metric="cosine")
        model.fit(X_train, y[train_idx])
        preds[test_idx] = model.predict(X_test)
    return preds


def cross_val_predict_gb(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for train_idx, test_idx in kf.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed
        )
        model.fit(X_train, y[train_idx])
        preds[test_idx] = model.predict(X_test)
    return preds


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deep-feature baseline for perceptual attributes")
    parser.add_argument("--perception", required=True, help="Perception config key (e.g., memorability, vc)")
    parser.add_argument("--extractor", default="vgg", choices=["vgg", "lpips", "clip"], help="Feature extractor")
    parser.add_argument("--kfold", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--knn-k", type=int, default=5, help="k for k-NN")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force-refetch", action="store_true", help="Re-download images even if cached")
    args = parser.parse_args()

    cfg = load_config(args.perception)
    df = load_data(cfg)
    print(f"[{args.perception}] {len(df)} images with ground-truth scores")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fetch images
    image_paths = []
    rows_kept = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching images"):
        name = row[cfg["image_col"]]
        url = row[cfg["url_col"]]
        cache = cached_image_path(name)
        if args.force_refetch and cache.exists():
            cache.unlink()
        path = get_or_fetch_image(name, url)
        if path is not None:
            image_paths.append(path)
            rows_kept.append(row)

    df = pd.DataFrame(rows_kept).reset_index(drop=True)
    y = df[cfg["gt_col"]].astype(float).values
    print(f"Successfully loaded {len(df)} images")

    # Extract embeddings, reusing a cached .npz if it covers every image
    # (avoids re-running the CNN forward pass on unchanged data).
    out_dir = RESULTS_DIR / args.perception
    cache_npz = out_dir / f"embeddings_{args.extractor}.npz"
    names = df[cfg["image_col"]].values
    X = None
    if cache_npz.exists() and not args.force_refetch:
        cached = np.load(cache_npz, allow_pickle=True)
        cached_names = list(cached["image_names"])
        if list(names) == cached_names:
            X = cached["embeddings"]
            print(f"Loaded cached {args.extractor.upper()} embeddings from {cache_npz} (shape {X.shape})")
        else:
            name_to_idx = {n: i for i, n in enumerate(cached_names)}
            if all(n in name_to_idx for n in names):
                X = cached["embeddings"][[name_to_idx[n] for n in names]]
                print(f"Loaded cached {args.extractor.upper()} embeddings from {cache_npz} (reordered, shape {X.shape})")

    if X is None:
        extractor = build_extractor(args.extractor, device)
        print(f"Extracting {args.extractor.upper()} embeddings...")
        X = extractor(image_paths)
        print(f"Embedding shape: {X.shape}")

    # Predict
    print("Running Ridge regression...")
    ridge_preds = cross_val_predict_ridge(X, y, args.kfold, args.seed)
    print(f"Running {args.knn_k}-NN...")
    knn_preds = cross_val_predict_knn(X, y, args.kfold, args.seed, k=args.knn_k)
    print("Running Gradient Boosting...")
    gb_preds = cross_val_predict_gb(X, y, args.kfold, args.seed)

    ridge_metrics = evaluate(y, ridge_preds)
    knn_metrics = evaluate(y, knn_preds)
    gb_metrics = evaluate(y, gb_preds)

    print("\nRidge regression metrics:")
    for k, v in ridge_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nk-NN metrics:")
    for k, v in knn_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nGradient Boosting metrics:")
    for k, v in gb_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save outputs
    out_dir = RESULTS_DIR / args.perception
    out_dir.mkdir(parents=True, exist_ok=True)

    df_out = df[[cfg["image_col"], cfg["gt_col"]]].copy()
    df_out["ridge_pred"] = ridge_preds
    df_out["knn_pred"] = knn_preds
    df_out["gb_pred"] = gb_preds
    out_csv = out_dir / f"deep_feature_baseline_{args.extractor}.csv"
    df_out.to_csv(out_csv, index=False)

    metrics = {
        "perception": args.perception,
        "extractor": args.extractor,
        "n_images": len(df),
        "embedding_dim": int(X.shape[1]),
        "ridge": ridge_metrics,
        "knn": knn_metrics,
        "gb": gb_metrics,
    }
    out_json = out_dir / f"deep_feature_metrics_{args.extractor}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    out_npz = out_dir / f"embeddings_{args.extractor}.npz"
    np.savez(out_npz, embeddings=X, image_names=df[cfg["image_col"]].values)

    print(f"\nSaved predictions to {out_csv}")
    print(f"Saved metrics to {out_json}")
    print(f"Saved embeddings to {out_npz}")


if __name__ == "__main__":
    main()
