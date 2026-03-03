"""
Generate Bird's-Eye-View (BEV) semantic + geometric targets.

Only cells visible within the front camera FOV are populated.
Each cell stores: semantic class, mean height, and slope.

Output .npz keys:
  - bev_semantics : [100, 100] uint8   – class index (0 = unobserved)
  - bev_height    : [100, 100] float32 – mean ground height (m, camera Y)
  - bev_slope     : [100, 100] float32 – local slope in degrees
  - bev_valid     : [100, 100] bool    – True where data was obtained
  - bev_fov_mask  : [100, 100] bool    – True for cells inside camera FOV
"""

import os
import glob
import numpy as np
from scipy import ndimage

# ── Configuration ───────────────────────────────────────────────────
HOME = os.path.expanduser("~")
DATASET_DIR = os.path.join(HOME, "Downloads", "Sample Dataset With Semantic Annotations")

SEMANTICS_DIR = os.path.join(DATASET_DIR, "pylon_camera_node", "semantics", "0000")
LIDAR_DIR     = os.path.join(DATASET_DIR, "os1_cloud_node_npy", "0000")
OUTPUT_DIR    = os.path.join(DATASET_DIR, "bev_targets", "0000")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# BEV Grid
BEV_RES    = 100             # 100×100 cells
NUM_CLASSES    = 13          # classes 0–12
UNOBSERVED     = 255         # sentinel value (class 0 = dirt road, so can't use 0)

# Physical extents in camera frame (metres)
Z_MIN, Z_MAX =  0.0, 25.0   # forward
X_MIN, X_MAX = -12.5, 12.5  # lateral
CELL_X = (X_MAX - X_MIN) / BEV_RES
CELL_Z = (Z_MAX - Z_MIN) / BEV_RES

# Height band (camera Y-axis, positive = downward)
Y_MIN, Y_MAX = -2.0, 3.0

# Camera intrinsics – adjusted from 1440×1080 original to 1280×720
# Original: FX=1037.350 FY=1124.614 CX=708.762 CY=549.905
# Horizontal: center-crop 1440→1280 (remove 80 px each side) → CX shifts, FX unchanged
# Vertical: scale 1080→720 (×2/3) → FY and CY scale
_CROP_X = (1440 - 1280) / 2   # 80 px
_SY     = 720.0 / 1080.0      # 2/3
FX  = 1037.350                 # unchanged (crop, not resize)
FY  = 1124.614 * _SY           # 749.743
CX  = 708.762  - _CROP_X       # 628.762
CY  = 549.905  * _SY           # 366.603
IMG_H, IMG_W = 720, 1280

# LiDAR → Camera extrinsic
T_VELO_CAM = np.array([
    [ 0.0, -1.0,  0.0,  0.2   ],
    [ 0.0,  0.0, -1.0,  0.275 ],
    [ 1.0,  0.0,  0.0, -0.355 ],
    [ 0.0,  0.0,  0.0,  1.0   ]
], dtype=np.float32)

INFILL_RADIUS = BEV_RES  # fill ALL FOV cells with nearest observed value


# ── Pre-compute camera FOV mask on the BEV grid ─────────────────────

def build_fov_mask():
    """For each BEV cell centre, check if it projects inside the image.
    
    The FOV forms a trapezoid: narrow near the vehicle, wide far away.
    We check horizontal projection only (vertical depends on ground height,
    but the horizontal FOV is the main constraint).
    """
    mask = np.zeros((BEV_RES, BEV_RES), dtype=bool)
    for r in range(BEV_RES):
        for c in range(BEV_RES):
            # Cell centre in camera frame (X_c, Z_c)
            x_c = X_MIN + (c + 0.5) * CELL_X
            z_c = Z_MIN + ((BEV_RES - 1 - r) + 0.5) * CELL_Z  # row 0 = farthest
            if z_c <= 0.1:
                continue
            # Horizontal image coordinate
            u = x_c * FX / z_c + CX
            if 0 <= u < IMG_W:
                mask[r, c] = True
    return mask

FOV_MASK = build_fov_mask()
print(f"FOV mask: {FOV_MASK.sum()} / {BEV_RES*BEV_RES} cells "
      f"({100*FOV_MASK.sum()/(BEV_RES*BEV_RES):.1f}%) inside camera FOV")


# ── Helpers ──────────────────────────────────────────────────────────

def compute_slope(h, valid, cx, cz):
    """Slope in degrees from smoothed height gradient."""
    hf = np.where(valid, h, 0.0).astype(np.float64)
    w  = valid.astype(np.float64)
    from scipy.ndimage import uniform_filter
    hn = uniform_filter(hf, size=3)
    wn = uniform_filter(w,  size=3)
    g  = wn > 0
    hs = np.zeros_like(hf)
    hs[g] = hn[g] / wn[g]
    dz, dx = np.gradient(hs, cz, cx)
    s = np.degrees(np.arctan(np.sqrt(dx**2 + dz**2))).astype(np.float32)
    s[~valid] = 0.0
    return s


def infill_nearest(grid, valid, radius):
    """NN infill within `radius` cells."""
    if valid.all() or not valid.any():
        return grid
    dist, idx = ndimage.distance_transform_edt(~valid, return_indices=True)
    fill = (~valid) & (dist <= radius)
    out = grid.copy()
    out[fill] = grid[idx[0][fill], idx[1][fill]]
    return out


# ── Main ─────────────────────────────────────────────────────────────

def generate_bev_targets():
    sem_files   = sorted(glob.glob(os.path.join(SEMANTICS_DIR, "*.npy")))
    lidar_files = sorted(glob.glob(os.path.join(LIDAR_DIR, "*.npy")))
    if not sem_files:
        print(f"No semantic maps in {SEMANTICS_DIR}"); return
    if not lidar_files:
        print(f"No LiDAR files in {LIDAR_DIR}"); return

    lidar_times, lidar_paths = [], []
    for lf in lidar_files:
        bn = os.path.splitext(os.path.basename(lf))[0]
        try:
            lidar_times.append(int(bn))
            lidar_paths.append(lf)
        except ValueError:
            pass
    lidar_times = np.array(lidar_times, dtype=np.int64)

    print(f"Found {len(sem_files)} semantic frames, {len(lidar_paths)} LiDAR sweeps.")
    print(f"BEV {BEV_RES}×{BEV_RES}, cell {CELL_X:.2f}×{CELL_Z:.2f} m")
    generated = 0

    for sem_path in sem_files:
        name = os.path.splitext(os.path.basename(sem_path))[0]
        try:
            img_time = int(name)
        except ValueError:
            continue

        diffs = np.abs(lidar_times - img_time)
        best  = np.argmin(diffs)
        if diffs[best] > 100_000_000:
            continue

        semantics = np.load(sem_path)             # [720, 1280] uint8
        points    = np.load(lidar_paths[best])    # [N, 4]
        xyz_velo  = points[:, :3].astype(np.float32)

        # LiDAR → Camera
        N = xyz_velo.shape[0]
        xyz_hom = np.hstack((xyz_velo, np.ones((N, 1), dtype=np.float32)))
        xyz_cam = (T_VELO_CAM @ xyz_hom.T).T[:, :3]
        X_c, Y_c, Z_c = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]

        # Spatial filter + BEV extents
        m = ((Z_c > 0.1) & (Y_c >= Y_MIN) & (Y_c <= Y_MAX) &
             (Z_c >= Z_MIN) & (Z_c <= Z_MAX) &
             (X_c >= X_MIN) & (X_c <= X_MAX))
        X_c, Y_c, Z_c = X_c[m], Y_c[m], Z_c[m]
        if X_c.size == 0:
            continue

        # Project to image – only keep points inside camera FOV
        u = (X_c * FX / Z_c + CX).astype(np.int32)
        v = (Y_c * FY / Z_c + CY).astype(np.int32)
        in_img = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        u, v = u[in_img], v[in_img]
        X_c, Y_c, Z_c = X_c[in_img], Y_c[in_img], Z_c[in_img]

        if X_c.size == 0:
            continue

        sem_labels = semantics[v, u]

        # BEV cell indices
        col = np.clip(
            np.floor((X_c - X_MIN) / (X_MAX - X_MIN) * BEV_RES).astype(np.int32),
            0, BEV_RES - 1)
        row = np.clip(
            np.floor((Z_c - Z_MIN) / (Z_MAX - Z_MIN) * BEV_RES).astype(np.int32),
            0, BEV_RES - 1)
        row = (BEV_RES - 1) - row  # flip: row-0 = farthest

        # Init grids
        bev_sem   = np.full((BEV_RES, BEV_RES), UNOBSERVED, dtype=np.uint8)
        bev_valid = np.zeros((BEV_RES, BEV_RES), dtype=bool)
        h_sum     = np.zeros((BEV_RES, BEV_RES), dtype=np.float64)
        h_cnt     = np.zeros((BEV_RES, BEV_RES), dtype=np.int32)

        cell_id = row * BEV_RES + col
        np.add.at(h_sum.ravel(), cell_id, Y_c.astype(np.float64))
        np.add.at(h_cnt.ravel(), cell_id, 1)

        # Majority vote for semantics
        for cid in np.unique(cell_id):
            pts = cell_id == cid
            r_, c_ = divmod(int(cid), BEV_RES)
            labels = sem_labels[pts]
            counts = np.bincount(labels, minlength=NUM_CLASSES)
            bev_sem[r_, c_]   = counts.argmax()
            bev_valid[r_, c_] = True

        # Height
        bev_height = np.zeros((BEV_RES, BEV_RES), dtype=np.float32)
        has_h = h_cnt > 0
        bev_height[has_h] = (h_sum[has_h] / h_cnt[has_h]).astype(np.float32)

        # Apply FOV mask – zero out anything outside camera FOV
        bev_sem[~FOV_MASK]    = UNOBSERVED
        bev_valid[~FOV_MASK]  = False
        bev_height[~FOV_MASK] = 0.0

        # Infill within FOV only
        bev_sem_filled = infill_nearest(bev_sem, bev_valid, INFILL_RADIUS)
        bev_sem_filled[~FOV_MASK] = UNOBSERVED  # re-mask after infill
        bev_height_filled = infill_nearest(bev_height, bev_valid, INFILL_RADIUS)
        bev_height_filled[~FOV_MASK] = 0.0

        # Slope from height
        bev_slope = compute_slope(bev_height_filled, bev_valid, CELL_X, CELL_Z)
        bev_slope[~FOV_MASK] = 0.0

        # Save
        out_path = os.path.join(OUTPUT_DIR, f"{name}.npz")
        np.savez_compressed(
            out_path,
            bev_semantics = bev_sem_filled,   # [100,100] uint8
            bev_height    = bev_height_filled, # [100,100] float32
            bev_slope     = bev_slope,         # [100,100] float32
            bev_valid     = bev_valid,         # [100,100] bool
            bev_fov_mask  = FOV_MASK,          # [100,100] bool
        )
        generated += 1
        if generated % 100 == 0:
            print(f"  ... {generated} frames  ({bev_valid.sum()} valid cells)")

    print(f"Finished – {generated} BEV targets in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_bev_targets()
