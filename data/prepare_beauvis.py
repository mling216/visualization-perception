"""
Prepare BeauVis image-level ground-truth data for LLM scoring.

Run from repo root:
    python data/prepare_beauvis.py

Expected source (cloned separately):
    _external/beauvis/

Outputs:
    data/beauvis_15.csv
    data/beauvis_items_15.csv

Columns in beauvis_15.csv:
    imageName, imageURL, gt_score_raw, gt_score, n_participants

Where:
    gt_score_raw = per-image mean of the 5 BeauVis items on the original 1-7 scale
    gt_score     = min-max normalized gt_score_raw to 0-1 (for compatibility with
                   existing scoring/evaluation scripts)
"""

from pathlib import Path
from typing import Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
BEAUVIS_ROOT = REPO_ROOT / "_external" / "beauvis"
EFA_DATA = BEAUVIS_ROOT / "03_EFA" / "data"
EFA_IMAGES = BEAUVIS_ROOT / "03_EFA" / "images"

OUT_MAIN = REPO_ROOT / "data" / "beauvis_15.csv"
OUT_ITEMS = REPO_ROOT / "data" / "beauvis_items_15.csv"

BEAUVIS_ITEMS = ["enjoyable", "likable", "pleasing", "nice", "appealing"]


def parse_item_column(col_name: str) -> Optional[str]:
    """Extract item name from columns like vis01[appealing]."""
    if "[" not in col_name or "]" not in col_name:
        return None
    return col_name.split("[", 1)[1].split("]", 1)[0].strip().lower()


def main() -> None:
    if not EFA_DATA.exists() or not EFA_IMAGES.exists():
        raise FileNotFoundError(
            "BeauVis source not found. Clone first:\n"
            "  git clone https://github.com/tingying-he/beauvis.git _external/beauvis"
        )

    rows = []
    item_rows = []

    for csv_path in sorted(EFA_DATA.glob("vis*.csv")):
        image_name = f"{csv_path.stem}.png"
        image_path = EFA_IMAGES / image_name
        if not image_path.exists():
            print(f"WARNING: missing image for {csv_path.name}: {image_name}")
            continue

        df = pd.read_csv(csv_path)
        item_map = {c: parse_item_column(c) for c in df.columns}
        keep_cols = [c for c, item in item_map.items() if item in BEAUVIS_ITEMS]

        if len(keep_cols) != len(BEAUVIS_ITEMS):
            found = sorted({item_map[c] for c in keep_cols})
            print(f"WARNING: expected 5 BeauVis items in {csv_path.name}, found: {found}")

        item_df = df[keep_cols].copy()
        item_df.columns = [item_map[c] for c in keep_cols]

        # Image-level BeauVis score: participant-wise mean across 5 items, then mean across participants.
        participant_means = item_df.mean(axis=1, skipna=True)
        gt_raw = float(participant_means.mean(skipna=True))

        rows.append(
            {
                "imageName": image_name,
                "imageURL": image_path.resolve().relative_to(REPO_ROOT).as_posix(),
                "gt_score_raw": gt_raw,
                "n_participants": int(len(item_df)),
            }
        )

        item_means = item_df.mean(axis=0, skipna=True).to_dict()
        item_rows.append(
            {
                "imageName": image_name,
                **{k: float(item_means.get(k, float("nan"))) for k in BEAUVIS_ITEMS},
            }
        )

    out = pd.DataFrame(rows).sort_values("imageName").reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No BeauVis rows were produced. Check source data paths.")

    lo = out["gt_score_raw"].min()
    hi = out["gt_score_raw"].max()
    if hi > lo:
        out["gt_score"] = (out["gt_score_raw"] - lo) / (hi - lo)
    else:
        out["gt_score"] = 0.0

    out = out[["imageName", "imageURL", "gt_score_raw", "gt_score", "n_participants"]]
    out.to_csv(OUT_MAIN, index=False)

    pd.DataFrame(item_rows).sort_values("imageName").to_csv(OUT_ITEMS, index=False)

    print(f"Wrote {len(out)} rows -> {OUT_MAIN}")
    print(
        "gt_score_raw range: "
        f"{out['gt_score_raw'].min():.3f} .. {out['gt_score_raw'].max():.3f}"
    )
    print("gt_score range: 0.000 .. 1.000")
    print(f"Wrote item means -> {OUT_ITEMS}")


if __name__ == "__main__":
    main()
