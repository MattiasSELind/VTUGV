"""
Visualise BEV semantic + geometric targets (100×100 FOV-limited grid).

Usage:
    python visualize_bev_targets.py                       # defaults
    python visualize_bev_targets.py --n 4 --start 200
"""

import os, sys, argparse, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Class definitions from estimate_semantics.py ─────────────────────
CLASS_NAMES = [
    "dirt road", "grass", "tree", "bush", "rock", "mud",
    "water", "sky", "vehicle", "person", "building", "fence", "terrain",
]
CLASS_COLORS = np.array([
    [160, 120,  80],  # 0  dirt road
    [124, 252,   0],  # 1  grass
    [ 34, 139,  34],  # 2  tree
    [  0, 180,   0],  # 3  bush
    [160, 160, 160],  # 4  rock
    [100,  70,  40],  # 5  mud
    [  0, 100, 255],  # 6  water
    [135, 206, 250],  # 7  sky
    [  0,   0, 180],  # 8  vehicle
    [255,   0,   0],  # 9  person
    [ 70,  70,  70],  # 10 building
    [190, 153, 153],  # 11 fence
    [210, 180, 140],  # 12 terrain
], dtype=np.uint8)

EXTENT = [-12.5, 12.5, 0, 25]
BEV_RES = 100

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default=os.path.join(
    os.path.expanduser("~"), "Downloads",
    "Sample Dataset With Semantic Annotations", "bev_targets", "0000"))
parser.add_argument("--n", type=int, default=4)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--save", default=None)
args = parser.parse_args()

files = sorted(glob.glob(os.path.join(args.dir, "*.npz")))
if not files: sys.exit(f"No .npz in {args.dir}")
files = files[args.start : args.start + args.n]
n = len(files)

sample = np.load(files[0])
print(f"Keys: {list(sample.keys())}")
for k in sample: print(f"  {k}: {sample[k].shape} {sample[k].dtype}")

legend_patches = [mpatches.Patch(color=CLASS_COLORS[i]/255., label=f"{i}: {CLASS_NAMES[i]}")
                  for i in range(len(CLASS_NAMES))]

# 4 columns: Semantics | Height | Slope | FOV + valid overlay
fig, axes = plt.subplots(n, 4, figsize=(20, 5*n), squeeze=False)
fig.suptitle("BEV Targets – 100×100 FOV-Limited Grid", fontsize=14, fontweight="bold")
col_titles = ["Semantics", "Height (m)", "Slope (°)", "Coverage (green=valid, gray=FOV)"]
for ax, t in zip(axes[0], col_titles):
    ax.set_title(t, fontsize=10, fontweight="bold")

def add_grid(ax):
    """Overlay light grid lines at every 10 cells (= 2.5 m)."""
    for x in np.linspace(EXTENT[0], EXTENT[1], 11):
        ax.axvline(x, color='white', linewidth=0.3, alpha=0.4)
    for y in np.linspace(EXTENT[2], EXTENT[3], 11):
        ax.axhline(y, color='white', linewidth=0.3, alpha=0.4)

for row, fp in enumerate(files):
    d = np.load(fp)
    fname = os.path.basename(fp)

    bev_sem  = d["bev_semantics"]
    bev_h    = d["bev_height"]
    bev_s    = d["bev_slope"]
    valid    = d["bev_valid"]
    fov      = d["bev_fov_mask"]

    # 1 – Semantics
    # Map 255 (unobserved) to a display color
    sem_display = bev_sem.copy()
    sem_display[sem_display == 255] = 0  # temp map for indexing
    rgb = CLASS_COLORS[sem_display].copy()
    rgb[~fov] = [0, 0, 0]                       # outside FOV = black
    rgb[bev_sem == 255] = [30, 30, 30]           # unobserved = dark gray
    axes[row,0].imshow(rgb, extent=EXTENT, aspect="equal", interpolation="nearest")
    axes[row,0].set_ylabel(fname, fontsize=8)
    add_grid(axes[row,0])

    # 2 – Height
    hm = bev_h.copy().astype(float)
    hm[~fov] = np.nan
    im_h = axes[row,1].imshow(hm, cmap="coolwarm", extent=EXTENT, aspect="equal",
                              interpolation="nearest")
    plt.colorbar(im_h, ax=axes[row,1], fraction=.04, pad=.04)
    add_grid(axes[row,1])

    # 3 – Slope
    sm = bev_s.copy().astype(float)
    sm[~fov] = np.nan
    im_s = axes[row,2].imshow(sm, cmap="YlOrRd", vmin=0, vmax=30,
                              extent=EXTENT, aspect="equal", interpolation="nearest")
    plt.colorbar(im_s, ax=axes[row,2], fraction=.04, pad=.04)
    add_grid(axes[row,2])

    # 4 – Coverage overlay: FOV in gray, valid in green
    overlay = np.zeros((BEV_RES, BEV_RES, 3), dtype=np.uint8)
    overlay[fov]  = [60, 60, 60]
    overlay[valid] = [0, 200, 0]
    axes[row,3].imshow(overlay, extent=EXTENT, aspect="equal", interpolation="nearest")
    pct = 100 * valid.sum() / max(fov.sum(), 1)
    axes[row,3].text(0, 24, f"{valid.sum()}/{fov.sum()} cells ({pct:.0f}%)",
                     ha='center', va='top', fontsize=8, color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    add_grid(axes[row,3])

for ax in axes.flat:
    ax.set_xlabel("Lateral (m)", fontsize=6)
    ax.tick_params(labelsize=5)

fig.legend(handles=legend_patches, loc="lower center",
           ncol=min(7, len(legend_patches)), fontsize=7,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()

if args.save:
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.save}")
else:
    plt.show()
