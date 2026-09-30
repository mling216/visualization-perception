"""Generate a 4x5 grid of sampled shared images, labeled with human scores."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
IMAGES = DATA / "image_cache"
FIGURES = ROOT / "figures"

ROWS, COLS = 4, 5
SEED = 42
CELL_PX = 600


def square_thumbnail(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((CELL_PX, CELL_PX), Image.LANCZOS)
    canvas = Image.new("RGB", (CELL_PX, CELL_PX), "white")
    canvas.paste(image, ((CELL_PX - image.width) // 2, (CELL_PX - image.height) // 2))
    return canvas


def main() -> None:
    vc = pd.read_csv(DATA / "vc_270.csv")[["imageName", "gt_score"]].rename(columns={"gt_score": "vc"})
    mem = pd.read_csv(DATA / "memorability_270.csv")[["imageName", "gt_score"]].rename(columns={"gt_score": "mem"})
    frame = vc.merge(mem, on="imageName", validate="one_to_one")
    sample = frame.sample(n=ROWS * COLS, random_state=SEED).sort_values("vc").reset_index(drop=True)

    fig, axes = plt.subplots(ROWS, COLS, figsize=(3.4, 3.2))
    for axis, row in zip(axes.ravel(), sample.itertuples()):
        axis.imshow(square_thumbnail(IMAGES / row.imageName))
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor("0.75")
            spine.set_linewidth(0.5)
        axis.set_xlabel(f"V {row.vc:.2f}, M {row.mem:.2f}", fontsize=5.5, labelpad=1.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.045, wspace=0.06, hspace=0.24)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "image_grid_sample.png", dpi=300)
    fig.savefig(FIGURES / "image_grid_sample.pdf", dpi=300)
    plt.close(fig)
    sample.to_csv(FIGURES / "image_grid_sample.csv", index=False)
    print(f"Wrote {FIGURES / 'image_grid_sample.pdf'}")


if __name__ == "__main__":
    main()
