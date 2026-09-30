# LLMs and Deep Features for Perceptual Assessment of Visualization Images

Code, prepared data, and results for the short paper

> Meng Ling and Xiyuan Lyu. **LLMs and Deep Features for Perceptual Assessment of Visualization Images.** 2nd Workshop on GenAI, Agents, and the Future of VIS (VISxGenAI), IEEE VIS 2026.

We ask GPT-5.4 and Claude Sonnet 4.6 to score visualization images for visual complexity, memorability, aesthetic pleasure, and readability, and compare their scores with published human ratings. As baselines, we fit Ridge regressions on frozen VGG19 features (a late-layer vector and an LPIPS-style multi-layer vector) and run ResMem, a memorability model trained on photographs, without refitting.

## What's here

| Folder | Contents |
|---|---|
| `config/` | Prompts and column mappings, one JSON file per attribute. |
| `data/` | Image lists with human scores, plus the scripts that build the BeauVis and PREVis tables. |
| `scripts/` | Scoring, baselines, analysis, and figures. |
| `results/` | Model outputs and analysis tables used in the paper. |
| `figures/` | Figures generated from `results/`. |

Source images are not included. The complexity and memorability tables point to image URLs; BeauVis and PREVis images come from the original authors' materials (see below). Please follow the licensing and citation terms of each source dataset.

## Datasets

| Attribute | Source | Images | File |
|---|---|---|---|
| Visual complexity | Chu et al., TVCG 2026 | 273 | `data/vc_270.csv` |
| Memorability | Borkin et al., TVCG 2013 (MASSVIS) | 273 | `data/memorability_270.csv` |
| Aesthetic pleasure | He et al., TVCG 2023 (BeauVis) | 15 | `data/beauvis_15.csv` |
| Readability | Cabouat et al., TVCG 2025 (PREVis) | 9 | `data/previs_all_9.csv` |

The complexity and memorability files cover the same 273 images, despite the `_270` in their names. `gt_chart_type` holds one harmonized set of chart-type labels; the two source papers name some types slightly differently.

To rebuild the BeauVis and PREVis tables, put the original materials under `_external/` (the expected folder names are at the top of `data/prepare_beauvis.py` and `data/prepare_previs.py`) and run both scripts. Their `imageURL` columns hold paths relative to the repo root.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

LLM scoring needs `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` in a local `.env` file.

## Where the paper's numbers come from

| Paper item | Script | Output |
|---|---|---|
| LLM correlations, run-level results, reliability, paired differences, chart-type breakdown, complexity–memorability correlation (Tables 2, 3, 6–8) | `analyze_three_llm_runs.py` | `results/three_run_analysis/` |
| Deep-feature correlations and bootstrap intervals (Tables 2, 5) | `deep_feature_baseline.py`, `analyze_three_llm_runs.py` | `results/vc270/`, `results/memorability270/`, `results/three_run_analysis/` |
| ResMem (Tables 2, 5) | `resmem_zeroshot.py` | `results/resmem_zeroshot_metrics.json` |
| Aesthetic pleasure and readability (Table 9) | `score_perception.py` | `results/beauvis/`, `results/previs/` |
| Fig. 1 and Fig. 3 (scatter plots) | `generate_prediction_scatter.py` | `figures/prediction_scatter_main.*`, `figures/prediction_scatter_exploratory.*` |
| Fig. 2 (sample images) | `generate_image_grid.py` | `figures/image_grid_sample.*` |

Table numbers refer to the camera-ready version. In the output files, "LPIPS + Ridge" is the paper's LPIPS-style baseline. `analyze_273_overlap.py` is an older single-run comparison and is listed with the exploratory analyses below.

## Reproducing the analysis

These steps use only the files in `results/` and need no API keys or images:

```bash
python scripts/analyze_three_llm_runs.py
python scripts/analyze_273_overlap.py
python scripts/generate_prediction_scatter.py
```

The bootstrap uses 5,000 paired image resamples with seed 42.

The steps below need the images. `deep_feature_baseline.py` downloads them into `data/image_cache/` on first use; the ResMem and image-grid scripts read from that cache.

```bash
python scripts/deep_feature_baseline.py --perception vc270 --extractor vgg
python scripts/deep_feature_baseline.py --perception vc270 --extractor lpips
python scripts/deep_feature_baseline.py --perception memorability270 --extractor vgg
python scripts/deep_feature_baseline.py --perception memorability270 --extractor lpips
python scripts/resmem_zeroshot.py
python scripts/generate_image_grid.py
```

## Rerunning the LLMs

```bash
python scripts/score_perception.py --perception vc --provider gpt                 # run 1
python scripts/score_perception.py --perception vc --provider gpt --run-id 2      # run 2
python scripts/score_perception.py --perception vc --provider gpt --run-id 3      # run 3
python scripts/aggregate_llm_runs.py --perception vc --model-tag gpt-5.4 --data-csv data/vc_270.csv
```

Use `--provider claude` for Claude Sonnet 4.6. For memorability, run 1 used `--perception memorability270` and runs 2–3 used `--perception memorability`. All calls use temperature 0, but hosted models still change over time, so new runs will not match the saved files exactly. The files in `results/` are the ones behind the paper.

Two things to know about the prompts:

- Besides the score, every prompt also asks the model to name the image's main chart type. We did not use those answers; the chart-type breakdown uses the dataset labels.
- The two memorability configs differ by one word in the opening line ("data visualization" in `memorability270.json`, "data visualization research" in `memorability.json`). Run 1 used the first and runs 2–3 the second; otherwise the prompts are identical.

## Other analyses

These scripts and outputs are exploratory and are not reported in the paper:

- `observer_convergence.py`, `analyze_observer_convergence.py`: whether LLM scores agree more with the deep-feature baselines than with humans (`results/three_run_analysis/observer_convergence*.csv`).
- `rsa_analysis.py`: pairwise LPIPS distances compared with score differences (`results/rsa_273_overlap.json`).
- `analyze_273_overlap.py` also writes `results/overlap_273_conflation.json`, which checks whether VGG-predicted complexity tracks LLM memorability scores.

## Citation

```bibtex
@inproceedings{Ling:2026:LDF,
  author    = {Meng Ling and Xiyuan Lyu},
  title     = {{LLMs} and Deep Features for Perceptual Assessment of Visualization Images},
  booktitle = {2nd Workshop on GenAI, Agents, and the Future of VIS (VISxGenAI), IEEE VIS},
  year      = {2026}
}
```
