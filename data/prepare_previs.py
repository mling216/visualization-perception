"""
Prepare PREVis readability datasets for LLM scoring.

Run from repo root:
    python data/prepare_previs.py

Required sources:
    _external/PREVis-scales/
    _external/PREVis-osf/phase2_stimuli/
    _external/PREVis-osf/phase3_stimuli/

Outputs:
    data/previs_validation_3.csv
    data/previs_development_6.csv
    data/previs_all_9.csv
    data/previs_images/validation/*.png
    data/previs_images/development/*.png

Ground-truth scoring logic:
    1) Compute 4 PREVis subscales from final validated items:
       - Understand: obvious, represent, understandEasi
       - Layout: messi, crowd, distract
       - DataRead: inform, identifi, find
       - DataFeat: visibl, see
    2) Composite readability = mean of the 4 subscales (equal weight per subscale).
    3) Aggregate per stimulus by mean across participants.
    4) Min-max normalize composite to [0, 1] inside each dataset.
"""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent

PREVIS_REPO = REPO_ROOT / "_external" / "PREVis-scales"
DEV_RATINGS = PREVIS_REPO / "research code" / "Replication - PREVis development figures" / "Data" / "ratings-stimulus.csv"
VAL_RATINGS = PREVIS_REPO / "research code" / "Replication - PREVis validation figures" / "Data" / "multigroup_ratings.csv"

PHASE2_PDFS = REPO_ROOT / "_external" / "PREVis-osf" / "phase2_stimuli"
PHASE3_PDFS = REPO_ROOT / "_external" / "PREVis-osf" / "phase3_stimuli"

OUT_IMG_DEV = REPO_ROOT / "data" / "previs_images" / "development"
OUT_IMG_VAL = REPO_ROOT / "data" / "previs_images" / "validation"

OUT_DEV = REPO_ROOT / "data" / "previs_development_6.csv"
OUT_VAL = REPO_ROOT / "data" / "previs_validation_3.csv"
OUT_ALL = REPO_ROOT / "data" / "previs_all_9.csv"


SCALES = {
    "Understand": ["obvious", "represent", "understandEasi"],
    "Layout": ["messi", "crowd", "distract"],
    "DataRead": ["inform", "identifi", "find"],
    "DataFeat": ["visibl", "see"],
}


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file/folder: {path}")


def pdf_to_png(pdf_path: Path, out_png: Path, zoom: float = 2.0) -> None:
    import fitz  # PyMuPDF

    out_png.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        # PyMuPDF API differs across versions: newer uses save(), older uses writePNG().
        if hasattr(pix, "save"):
            pix.save(str(out_png))
        else:
            pix.writePNG(str(out_png))
    finally:
        doc.close()


def compute_subscales(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for scale, cols in SCALES.items():
        missing = [c for c in cols if c not in out.columns]
        if missing:
            raise KeyError(f"Missing PREVis columns for {scale}: {missing}")
        out[scale] = out[cols].mean(axis=1)
    out["readability_raw"] = out[["Understand", "Layout", "DataRead", "DataFeat"]].mean(axis=1)
    return out


def normalize01(series: pd.Series) -> pd.Series:
    lo, hi = float(series.min()), float(series.max())
    if hi <= lo:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - lo) / (hi - lo)


def map_pdf_by_stimulus(folder: Path) -> dict:
    ensure_exists(folder)
    mapping = {}
    for pdf in sorted(folder.glob("*.pdf")):
        letter = pdf.stem[0].upper()
        if letter in "ABCDEF":
            mapping[letter] = pdf
    return mapping


def build_dataset(ratings_df: pd.DataFrame, pdf_map: dict, out_img_dir: Path, dataset_name: str) -> pd.DataFrame:
    work = ratings_df.copy()

    # Some source files include an unnamed index column from previous exports.
    unnamed = [c for c in work.columns if str(c).lower().startswith("unnamed:")]
    if unnamed:
        work = work.drop(columns=unnamed)

    if "stimulus" not in work.columns:
        raise KeyError("Expected a 'stimulus' column in PREVis ratings file")

    scored = compute_subscales(work)
    grouped = scored.groupby("stimulus").agg(
        {
            "readability_raw": ["size", "mean"],
            "Understand": "mean",
            "Layout": "mean",
            "DataRead": "mean",
            "DataFeat": "mean",
        }
    )
    grouped.columns = [
        "n_participants",
        "gt_score_raw",
        "Understand",
        "Layout",
        "DataRead",
        "DataFeat",
    ]
    grouped = grouped.reset_index()

    grouped["gt_score"] = normalize01(grouped["gt_score_raw"]) 

    rows = []
    for _, r in grouped.sort_values("stimulus").iterrows():
        stim = str(r["stimulus"]).strip().upper()
        if stim not in pdf_map:
            raise FileNotFoundError(f"No stimulus PDF found for '{stim}' in {dataset_name}")

        pdf_path = pdf_map[stim]
        png_name = f"previs_{dataset_name}_{stim}.png"
        png_path = out_img_dir / png_name
        if not png_path.exists():
            pdf_to_png(pdf_path, png_path)

        rows.append(
            {
                "imageName": png_name,
                "imageURL": png_path.resolve().relative_to(REPO_ROOT).as_posix(),
                "stimulus": stim,
                "gt_score_raw": float(r["gt_score_raw"]),
                "gt_score": float(r["gt_score"]),
                "n_participants": int(r["n_participants"]),
                "understand": float(r["Understand"]),
                "layout": float(r["Layout"]),
                "dataread": float(r["DataRead"]),
                "datafeat": float(r["DataFeat"]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ensure_exists(DEV_RATINGS)
    ensure_exists(VAL_RATINGS)

    dev_pdf_map = map_pdf_by_stimulus(PHASE2_PDFS)
    val_pdf_map = map_pdf_by_stimulus(PHASE3_PDFS)

    dev_df = pd.read_csv(DEV_RATINGS)
    val_df = pd.read_csv(VAL_RATINGS)

    out_dev = build_dataset(dev_df, dev_pdf_map, OUT_IMG_DEV, "development")
    out_val = build_dataset(val_df, val_pdf_map, OUT_IMG_VAL, "validation")

    out_dev["split"] = "development"
    out_val["split"] = "validation"
    out_dev = out_dev.rename(columns={"gt_score": "gt_score_split"})
    out_val = out_val.rename(columns={"gt_score": "gt_score_split"})
    out_all = pd.concat([out_dev, out_val], ignore_index=True, sort=False)
    out_all["gt_score"] = normalize01(out_all["gt_score_raw"])

    # Keep the same column order as other datasets, with extra metadata at the end.
    col_order = [
        "imageName",
        "imageURL",
        "stimulus",
        "gt_score_raw",
        "gt_score",
        "n_participants",
        "understand",
        "layout",
        "dataread",
        "datafeat",
        "split",
        "gt_score_split",
    ]
    out_all = out_all[col_order]

    # Restore gt_score naming in split-specific outputs for compatibility.
    out_dev = out_dev.rename(columns={"gt_score_split": "gt_score"})
    out_val = out_val.rename(columns={"gt_score_split": "gt_score"})

    out_dev.to_csv(OUT_DEV, index=False)
    out_val.to_csv(OUT_VAL, index=False)
    out_all.to_csv(OUT_ALL, index=False)

    print(f"Wrote {len(out_dev)} rows -> {OUT_DEV}")
    print(
        "  raw range: "
        f"{out_dev['gt_score_raw'].min():.3f} .. {out_dev['gt_score_raw'].max():.3f}"
    )
    print(f"Wrote {len(out_val)} rows -> {OUT_VAL}")
    print(
        "  raw range: "
        f"{out_val['gt_score_raw'].min():.3f} .. {out_val['gt_score_raw'].max():.3f}"
    )
    print(f"Wrote {len(out_all)} rows -> {OUT_ALL}")


if __name__ == "__main__":
    main()
