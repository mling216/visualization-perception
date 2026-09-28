"""Aggregate three repeated LLM scoring runs by image.

Run 1 is the existing canonical score file. Runs 2 and 3 are created with
score_perception.py --run-id 2 and --run-id 3.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load_run(path: Path, run_id: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing run {run_id}: {path}")
    frame = pd.read_csv(path)
    required = {"imageName", "score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["imageName"].duplicated().any():
        raise ValueError(f"{path} contains duplicate imageName values")
    frame = frame.copy()
    frame["run_id"] = run_id
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    return frame


def aggregate(perception: str, model_tag: str, data_csv: Path | None = None) -> Path:
    result_dir = RESULTS_DIR / perception
    run1_dir = result_dir
    run1_prefix = perception
    if perception == "memorability":
        run1_dir = RESULTS_DIR / "memorability270"
        run1_prefix = "memorability270"
    files = [
        run1_dir / f"{run1_prefix}_{model_tag}_scores.csv",
        result_dir / "runs" / f"{perception}_{model_tag}_run2_scores.csv",
        result_dir / "runs" / f"{perception}_{model_tag}_run3_scores.csv",
    ]
    expected_images = None
    if data_csv is not None:
        target = pd.read_csv(data_csv)
        if "imageName" not in target.columns:
            raise ValueError(f"{data_csv} must contain an imageName column")
        expected_images = set(target["imageName"].dropna())

    loaded_runs = []
    for run_id, path in enumerate(files, 1):
        frame = load_run(path, run_id)
        if expected_images is not None:
            frame = frame[frame["imageName"].isin(expected_images)].copy()
            missing = expected_images.difference(frame["imageName"])
            if missing:
                raise ValueError(
                    f"Run {run_id} is missing {len(missing)} target images; "
                    f"examples: {sorted(missing)[:5]}"
                )
        loaded_runs.append(frame)
    runs = pd.concat(loaded_runs, ignore_index=True)

    counts = runs.groupby("imageName")["run_id"].nunique()
    incomplete = counts[counts != 3]
    if not incomplete.empty:
        raise ValueError(
            f"Expected 3 runs for every image; incomplete images: {list(incomplete.index[:10])}"
        )

    def summarize(group: pd.DataFrame) -> pd.Series:
        scores = group["score"].dropna().to_numpy(dtype=float)
        if len(scores) != 3:
            return pd.Series({
                "score": np.nan,
                "score_sd": np.nan,
                "score_ci_low": np.nan,
                "score_ci_high": np.nan,
                "run_count": len(scores),
            })
        mean = float(np.mean(scores))
        sd = float(np.std(scores, ddof=1))
        margin = float(stats.t.ppf(0.975, df=2) * sd / np.sqrt(3))
        return pd.Series({
            "score": mean,
            "score_sd": sd,
            "score_ci_low": mean - margin,
            "score_ci_high": mean + margin,
            "run_count": 3,
        })

    summary = runs.groupby("imageName", sort=False).apply(summarize, include_groups=False).reset_index()
    metadata = runs.sort_values("run_id").drop_duplicates("imageName")
    metadata_columns = [
        column for column in ["imageName", "chart_type"] if column in metadata.columns
    ]
    if metadata_columns:
        summary = metadata[metadata_columns].merge(summary, on="imageName", how="right")

    confidence = (
        runs.groupby("imageName")["confidence"]
        .agg(confidence_mean="mean", confidence_sd="std")
        .reset_index()
        if "confidence" in runs.columns
        else None
    )
    if confidence is not None:
        summary = summary.merge(confidence, on="imageName", how="left")

    output = result_dir / "aggregated" / f"{perception}_{model_tag}_3run_mean_scores.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate three repeated LLM score files.")
    parser.add_argument("--perception", required=True)
    parser.add_argument("--model-tag", required=True, dest="model_tag")
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=None,
        help="Restrict aggregation to the imageName values in this target CSV",
    )
    args = parser.parse_args()
    data_csv = args.data_csv
    if data_csv is not None and not data_csv.is_absolute():
        data_csv = REPO_ROOT / data_csv
    output = aggregate(args.perception, args.model_tag, data_csv)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()