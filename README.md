# Machine Perception of Visualization

Code, prepared inputs, figures, and results for:

**Multimodal LLMs and Deep Features as Perceptual Observers of Abstract Visualizations**

## Contents

- `config/` - perception definitions and scoring configuration.
- `data/` - prepared input tables for visual complexity, memorability, BeauVis, and PREVis.
- `figures/` - generated figures.
- `results/` - model predictions, baselines, and analysis outputs.
- `scripts/` - scoring, feature extraction, analysis, and plotting scripts.
- `requirements.txt` - Python dependencies.

Recent ResMem/FalseResMem, FAR, HR, and UMAP experiments are not included in the public analysis folders.

Original source images are not redistributed. CSV files retain source identifiers and URLs where available; follow the source datasets' licensing and citation requirements.

## Setup

From the repository root:

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

## Reproduce analyses

Regenerate the prediction figure:

```bash
python scripts/generate_prediction_scatter.py
```

Regenerate the repeated-run analysis:

```bash
python scripts/analyze_three_llm_runs.py
```

This computes three-run means, 5,000 paired image-bootstrap intervals (seed 42), run-level reliability, score-band summaries, and paired model comparisons.

Compare model convergence with human alignment:

```bash
python scripts/analyze_observer_convergence.py
```

To rerun model scoring, provide access to the source images and set the provider API key in a local `.env` file. Example:

```bash
python scripts/score_perception.py --perception vc --provider gpt --limit 20
```

For deep-feature baselines:

```bash
python scripts/deep_feature_baseline.py --perception vc --extractor vgg
python scripts/deep_feature_baseline.py --perception vc --extractor lpips
```

API responses can change over time. The checked-in files under `results/` are the outputs used for the reported analyses.