"""
Visualise BEV semantic + geometric targets (50×50 FOV-limited grid).

Usage:
    python visualize_bev_targets.py                       # defaults
    python visualize_bev_targets.py --n 4 --start 200
"""

import os, sys, argparse, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Class definitions ─────────────────────────────────────────────────────────
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

EXTENT  = [-12.5, 12.5, 0, 25]
BEV_RES = 50

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default=os.path.join(
    os.path.expanduser("~"), "Downloads",
    "Sample Dataset With Semantic Annotations", "bev_targets", "0000"))
parser.add_argument("--n",     type=int, default=3)
parser.add_argument("--start", type=int, default=30)
parser.add_argument("--save",  default=None)
args = parser.parse_args()

files = sorted(glob.glob(os.path.join(args.dir, "*.npz")))
if not files:
    sys.exit(f"No .npz in {args.dir}")
files = files[args.start : args.start + args.n]
n = len(files)

sample = np.load(files[0])
print(f"Keys: {list(sample.keys())}")
for k in sample:
    print(f"  {k}: {sample[k].shape} {sample[k].dtype}")

has_sem       = "bev_semantics"  in sample
has_conf      = "bev_confidence" in sample
has_clearance = "bev_clearance"  in sample

# Build column list dynamically based on available keys
cols = ["Height (m)", "Slope (°)", "Clearance (m)"] if has_clearance else ["Height (m)", "Slope (°)"]
if has_conf:
    cols.append("Ground Pt Count")
if has_sem:
    cols.append("Semantics")
cols.append("Coverage")
n_cols = len(cols)

legend_patches = [mpatches.Patch(color=CLASS_COLORS[i] / 255., label=f"{i}: {CLASS_NAMES[i]}")
                  for i in range(len(CLASS_NAMES))]


def add_grid(ax):
    for x in np.linspace(EXTENT[0], EXTENT[1], 11):
        ax.axvline(x, color='white', linewidth=0.3, alpha=0.4)
    for y in np.linspace(EXTENT[2], EXTENT[3], 11):
        ax.axhline(y, color='white', linewidth=0.3, alpha=0.4)


fig, axes = plt.subplots(n, n_cols, figsize=(5 * n_cols, 5 * n), squeeze=False)
fig.suptitle(f"BEV Targets – {BEV_RES}×{BEV_RES} FOV-Limited Grid", fontsize=14, fontweight="bold")
for ax, t in zip(axes[0], cols):
    ax.set_title(t, fontsize=10, fontweight="bold")

for row, fp in enumerate(files):
    d     = np.load(fp)
    fname = os.path.basename(fp)
    fov   = d["bev_fov_mask"]
    valid = d["bev_valid"]

    col = 0

    # Height
    hm = d["bev_height"].astype(float)
    hm[~fov] = np.nan
    im = axes[row, col].imshow(hm, cmap="coolwarm", extent=EXTENT, aspect="equal", interpolation="nearest")
    axes[row, col].set_ylabel(fname, fontsize=8)
    plt.colorbar(im, ax=axes[row, col], fraction=.04, pad=.04)
    add_grid(axes[row, col])
    col += 1

    # Slope
    sm = d["bev_slope"].astype(float)
    sm[~fov] = np.nan
    im = axes[row, col].imshow(sm, cmap="YlOrRd", vmin=0, vmax=30, extent=EXTENT, aspect="equal", interpolation="nearest")
    plt.colorbar(im, ax=axes[row, col], fraction=.04, pad=.04)
    add_grid(axes[row, col])
    col += 1

    # Clearance
    if has_clearance:
        ag = d["bev_clearance"].astype(float)
        ag[~fov] = np.nan
        im = axes[row, col].imshow(ag, cmap="viridis", vmin=0, vmax=3, extent=EXTENT, aspect="equal", interpolation="nearest")
        plt.colorbar(im, ax=axes[row, col], fraction=.04, pad=.04)
        add_grid(axes[row, col])
        col += 1

    # Confidence
    if has_conf:
        im = axes[row, col].imshow(d["bev_confidence"], cmap="viridis", extent=EXTENT, aspect="equal", interpolation="nearest")
        plt.colorbar(im, ax=axes[row, col], fraction=.04, pad=.04)
        add_grid(axes[row, col])
        col += 1

    # Semantics
    if has_sem:
        sem = d["bev_semantics"]
        sem_display = np.clip(sem, 0, len(CLASS_COLORS) - 1)
        rgb = CLASS_COLORS[sem_display].copy()
        rgb[~fov]       = [0,  0,  0]    # outside FOV = black
        rgb[sem  == -1] = [30, 30, 30]   # unobserved  = dark gray
        rgb[sem == 255] = [30, 30, 30]   # legacy unobserved value
        axes[row, col].imshow(rgb, extent=EXTENT, aspect="equal", interpolation="nearest")
        add_grid(axes[row, col])
        col += 1

    # Coverage
    overlay = np.zeros((BEV_RES, BEV_RES, 3), dtype=np.uint8)
    overlay[fov]   = [60, 60, 60]
    overlay[valid] = [0, 200, 0]
    axes[row, col].imshow(overlay, extent=EXTENT, aspect="equal", interpolation="nearest")
    pct = 100 * valid.sum() / max(fov.sum(), 1)
    axes[row, col].text(0, 24, f"{valid.sum()}/{fov.sum()} cells ({pct:.0f}%)",
                        ha='center', va='top', fontsize=8, color='white',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    add_grid(axes[row, col])

for ax in axes.flat:
    ax.set_xlabel("Lateral (m)", fontsize=6)
    ax.tick_params(labelsize=5)

if has_sem:
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=min(7, len(legend_patches)), fontsize=7,
               bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()

if args.save:
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.save}")
else:
    plt.show()
