"""
Generate Hybrid BEV height + slope maps from accumulated LiDAR scans + Dense Relative Depth.

Pipeline:
  1. Accumulate LiDAR scans (e.g. +/-7) into ego frame using poses.csv.
  2. Load dense 16-bit metric depth map (already calibrated to LiDAR in generate_depth.py).
  3. Unproject dense depth to 3D, transform from camera to ego frame.
  4. Filter out dense depth points < 4m forward (to remove the 'ledge' artifact).
  5. Concatenate LiDAR and Depth 3D points.
  6. Bin into BEV grid (ego frame), compute height, slope, valid mask.
  7. Apply Camera FOV mask.
  8. Save .npz and optional visualisation PNGs.

Usage:
  python generate_bev_targets_hybrid.py
  python generate_bev_targets_hybrid.py --num_frames 10 --visualize
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import uniform_filter
from scipy import ndimage

# -- Configuration ------------------------------------------------------------
PC_DIR     = Path(r"C:\Users\matli\Desktop\Dataset Preprocessing\extracted\pointclouds_center")
DEPTH_DIR  = Path(r"C:\Users\matli\Desktop\Dataset Preprocessing\extracted\depth_maps")
POSE_CSV   = Path(r"C:\Users\matli\Desktop\Dataset Preprocessing\poses.csv")
IMU_CSV    = Path(r"C:\Users\matli\Desktop\Dataset Preprocessing\extracted\imu_ouster\imu.csv")
OUTPUT_DIR = Path(r"C:\Users\matli\Desktop\Dataset Preprocessing\extracted\bev_targets_hybrid")

# BEV grid
BEV_RES   = 50
X_MIN, X_MAX =  0.0, 25.0    # forward (m)
Y_MIN, Y_MAX = -12.5, 12.5   # lateral (m)
CELL_X = (X_MAX - X_MIN) / BEV_RES
CELL_Y = (Y_MAX - Y_MIN) / BEV_RES

# Accumulation
ACCUM_HALF_WINDOW = 7

# Point filtering
MIN_RANGE_LIDAR = 1.0         # m
MAX_RANGE_LIDAR = 50.0        # m
# Filter out depth points less than this distance forward to remove ledge artifact
MIN_FORWARD_DEPTH = 3.5       # m (in ego frame X)
MAX_RANGE_DEPTH = 50.0        # m

Z_BAND_MIN = -3.0
Z_BAND_MAX =  5.0

# BEV cell thresholds
MIN_PTS_PER_CELL = 3
HEIGHT_PERCENTILE = 10
INFILL_RADIUS = 7

LIDAR_HZ = 10.0

# -- Camera intrinsics & FOV constraints --------------------------
FX, FY = 897.929, 974.238
CX, CY = 648.650, 256.277
IMG_H, IMG_W = 720, 1280
PIXEL_STRIDE = 2 # skip pixels in depth map to speed up

# LiDAR -> Camera transform
T_cam_lidar = np.array([
    [ 0.0, -1.0,  0.0, -0.2   ],
    [ 0.0,  0.0, -1.0, -0.275 ],
    [ 1.0,  0.0,  0.0, -0.355 ],
    [ 0.0,  0.0,  0.0,  1.0   ],
], dtype=np.float64)
T_lidar_cam = np.linalg.inv(T_cam_lidar)

_uu = np.arange(0, IMG_W, PIXEL_STRIDE, dtype=np.float32)
_vv = np.arange(0, IMG_H, PIXEL_STRIDE, dtype=np.float32)
_grid_u, _grid_v = np.meshgrid(_uu, _vv)
_ray_x = (_grid_u - CX) / FX
_ray_y = (_grid_v - CY) / FY

def build_fov_mask():
    mask = np.zeros((BEV_RES, BEV_RES), dtype=bool)
    for r in range(BEV_RES):
        for c in range(BEV_RES):
            x_lidar = X_MIN + (BEV_RES - 1 - r + 0.5) * CELL_X
            y_lidar = Y_MIN + (c + 0.5) * CELL_Y
            z_lidar = 0.0

            pt_lidar = np.array([x_lidar, y_lidar, z_lidar, 1.0])
            pt_cam = T_cam_lidar @ pt_lidar
            z_c = pt_cam[2]
            
            if z_c <= 0.1:
                continue
            
            u = pt_cam[0] * FX / z_c + CX
            v = pt_cam[1] * FY / z_c + CY
            
            if 0 <= u < IMG_W and 0 <= v < IMG_H:
                mask[r, c] = True
    return mask

FOV_MASK = build_fov_mask()


# -- Pose loading -------------------------------------------------------------
def load_poses(csv_path):
    print(f"Loading poses from {csv_path} ...")
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    ts  = data[:, 0] * 1e-9
    pos = data[:, 1:4]
    quat = data[:, 4:8]
    return ts, pos, quat

def get_imu_t0(imu_csv):
    with open(imu_csv, "r") as f:
        f.readline()
        return float(f.readline().split(",")[0])

def build_pose_interpolator(pose_ts, pos, quat):
    _, unique_idx = np.unique(pose_ts, return_index=True)
    unique_idx = np.sort(unique_idx)
    ts_u  = pose_ts[unique_idx]
    pos_u = pos[unique_idx]
    quat_u = quat[unique_idx]
    rots = Rotation.from_quat(quat_u)
    slerp = Slerp(ts_u, rots)

    def interp_pose(t):
        t_clamped = np.clip(t, ts_u[0], ts_u[-1])
        px = np.interp(t_clamped, ts_u, pos_u[:, 0])
        py = np.interp(t_clamped, ts_u, pos_u[:, 1])
        pz = np.interp(t_clamped, ts_u, pos_u[:, 2])
        rot = slerp(t_clamped)
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3]  = [px, py, pz]
        return T
    return interp_pose


# -- Data loading -------------------------------------------------------------
def load_scan(pc_path):
    pts = np.load(pc_path, allow_pickle=True)
    x, y, z = pts['x'].astype(np.float64), pts['y'].astype(np.float64), pts['z'].astype(np.float64)
    rng = pts['range']
    valid = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        & (rng > 0)
        & (np.abs(x) < 200) & (np.abs(y) < 200) & (np.abs(z) < 200)
        & ~(
            ((np.abs(x) < 0.001) & (np.abs(y) < 0.001))
            | ((np.abs(x) < 0.001) & (np.abs(z) < 0.001))
            | ((np.abs(y) < 0.001) & (np.abs(z) < 0.001))
        )
    )
    xyz = np.stack([x[valid], y[valid], z[valid]], axis=1)
    r = np.linalg.norm(xyz, axis=1)
    mask = (r >= MIN_RANGE_LIDAR) & (r <= MAX_RANGE_LIDAR)
    return xyz[mask]

def load_depth_points(depth_path):
    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        return np.zeros((0, 3))
    
    depth_sub = depth_img[::PIXEL_STRIDE, ::PIXEL_STRIDE].astype(np.float32)
    depth_m = depth_sub / 1000.0
    
    valid_depth = (depth_m > MIN_RANGE_LIDAR) & (depth_m < MAX_RANGE_DEPTH)
    
    Z_c = depth_m[valid_depth]
    X_c = _ray_x[valid_depth] * Z_c
    Y_c = _ray_y[valid_depth] * Z_c
    
    pts_cam = np.stack([X_c, Y_c, Z_c], axis=1) # (N, 3)
    
    # Transform to ego frame
    ones = np.ones((len(pts_cam), 1))
    pts_cam_h = np.hstack([pts_cam, ones])
    pts_ego = (T_lidar_cam @ pts_cam_h.T).T[:, :3]
    
    # Filter ledge artifact (forward < MIN_FORWARD_DEPTH)
    # the ledge usually appears right at the bottom of the camera frame
    valid_ledge = pts_ego[:, 0] >= MIN_FORWARD_DEPTH
    
    return pts_ego[valid_ledge]


# -- BEV helpers --------------------------------------------------------------
def compute_slope(h, valid):
    hf = np.where(valid, h, 0.0).astype(np.float64)
    w  = valid.astype(np.float64)
    hn = uniform_filter(hf, size=3)
    wn = uniform_filter(w, size=3)
    g  = wn > 0
    hs = np.zeros_like(hf)
    hs[g] = hn[g] / wn[g]
    dy, dx = np.gradient(hs, CELL_X, CELL_Y)
    s = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)
    s[~valid] = 0.0
    return s

def infill_nearest(grid, valid, radius):
    if valid.all() or not valid.any():
        return grid
    dist, idx = ndimage.distance_transform_edt(~valid, return_indices=True)
    fill = (~valid) & (dist <= radius)
    out = grid.copy()
    out[fill] = grid[idx[0][fill], idx[1][fill]]
    return out

def points_to_bev(pts_ego, sem_img=None, feat_map=None):
    x, y, z = pts_ego[:, 0], pts_ego[:, 1], pts_ego[:, 2]

    # Filter bounds
    mask = (
        (x >= X_MIN) & (x < X_MAX) &
        (y >= Y_MIN) & (y < Y_MAX) &
        (z >= Z_BAND_MIN) & (z < Z_BAND_MAX)
    )
    x, y, z = x[mask], y[mask], z[mask]
    pts_ego_valid = pts_ego[mask]

    bev_sem = np.full((BEV_RES, BEV_RES), -1, dtype=np.int32) if sem_img is not None else None
    bev_feat = np.zeros((BEV_RES, BEV_RES, feat_map.shape[-1]), dtype=np.float32) if feat_map is not None else None

    if len(x) == 0:
        return (np.zeros((BEV_RES, BEV_RES), dtype=np.float32),
                np.zeros((BEV_RES, BEV_RES), dtype=np.float32),
                np.zeros((BEV_RES, BEV_RES), dtype=bool),
                bev_sem, bev_feat)

    # Project valid ego points to camera image to get (u, v) mapping
    if sem_img is not None or feat_map is not None:
        ones = np.ones((len(pts_ego_valid), 1))
        pts_cam = (T_cam_lidar @ np.hstack([pts_ego_valid, ones]).T).T
        
        front_mask = pts_cam[:, 2] > 0.1
        
        u = np.zeros(len(pts_cam), dtype=int)
        v = np.zeros(len(pts_cam), dtype=int)
        valid_uv = np.zeros(len(pts_cam), dtype=bool)
        
        if np.any(front_mask):
            z_c = pts_cam[front_mask, 2]
            u_c = (pts_cam[front_mask, 0] * FX / z_c + CX).astype(int)
            v_c = (pts_cam[front_mask, 1] * FY / z_c + CY).astype(int)
            
            in_img = (u_c >= 0) & (u_c < IMG_W) & (v_c >= 0) & (v_c < IMG_H)
            
            valid_idx = np.where(front_mask)[0][in_img]
            u[valid_idx] = u_c[in_img]
            v[valid_idx] = v_c[in_img]
            valid_uv[valid_idx] = True

        pt_sem = np.full(len(pts_ego_valid), -1, dtype=np.int32)
        if sem_img is not None:
            pt_sem[valid_uv] = sem_img[v[valid_uv], u[valid_uv]]
            
        if feat_map is not None:
            pt_feat = np.zeros((len(pts_ego_valid), feat_map.shape[-1]), dtype=np.float32)
            pt_feat[valid_uv] = feat_map[v[valid_uv], u[valid_uv]]

    # To match Camera BEV (col 0 = left = X_MIN = LiDAR Y_MAX)
    # y=Y_MAX -> col=0. y=Y_MIN -> col=99
    col = BEV_RES - 1 - np.floor((y - Y_MIN) / CELL_Y).astype(int)
    row = BEV_RES - 1 - np.floor((x - X_MIN) / CELL_X).astype(int)
    col = np.clip(col, 0, BEV_RES - 1)
    row = np.clip(row, 0, BEV_RES - 1)

    bev_height = np.full((BEV_RES, BEV_RES), np.nan, dtype=np.float32)
    bev_valid  = np.zeros((BEV_RES, BEV_RES), dtype=bool)

    cell_idx = row * BEV_RES + col
    order = np.argsort(cell_idx)
    cell_idx_sorted = cell_idx[order]
    z_sorted = z[order]

    boundaries = np.searchsorted(cell_idx_sorted, np.arange(BEV_RES * BEV_RES))
    boundaries = np.append(boundaries, len(cell_idx_sorted))

    pt_sem_sorted = pt_sem[order] if sem_img is not None else None
    pt_feat_sorted = pt_feat[order] if feat_map is not None else None
    valid_uv_sorted = valid_uv[order] if (sem_img is not None or feat_map is not None) else None

    for cell in range(BEV_RES * BEV_RES):
        start, end = boundaries[cell], boundaries[cell + 1]
        n = end - start
        if n >= MIN_PTS_PER_CELL:
            r_idx = cell // BEV_RES
            c_idx = cell % BEV_RES
            cell_z = z_sorted[start:end]
            bev_height[r_idx, c_idx] = np.percentile(cell_z, HEIGHT_PERCENTILE)
            bev_valid[r_idx, c_idx] = True
            
            if sem_img is not None:
                c_sem = pt_sem_sorted[start:end]
                c_sem_valid = c_sem[c_sem >= 0]
                if len(c_sem_valid) > 0:
                    vals, counts = np.unique(c_sem_valid, return_counts=True)
                    bev_sem[r_idx, c_idx] = vals[np.argmax(counts)]
                    
            if feat_map is not None:
                c_feat = pt_feat_sorted[start:end]
                c_valid = valid_uv_sorted[start:end]
                if np.any(c_valid):
                    bev_feat[r_idx, c_idx] = np.mean(c_feat[c_valid], axis=0)

    bev_height_filled = infill_nearest(
        np.where(bev_valid, bev_height, 0.0).astype(np.float32),
        bev_valid, INFILL_RADIUS
    )
    valid_filled = bev_valid | (infill_nearest(
        bev_valid.astype(np.float32), bev_valid, INFILL_RADIUS
    ) > 0)
    bev_height_filled[~valid_filled] = 0.0

    bev_height_filled[~FOV_MASK] = 0.0
    valid_filled &= FOV_MASK
    bev_slope = compute_slope(bev_height_filled, valid_filled)

    return bev_height_filled, bev_slope, valid_filled, bev_sem, bev_feat


# -- Main pipeline ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sem_dir", type=str, default="")
    parser.add_argument("--feat_dir", type=str, default="")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        (OUTPUT_DIR / "viz").mkdir(exist_ok=True)

    scan_files = sorted(f for f in os.listdir(PC_DIR) if f.endswith(".npy"))
    # The depth maps are named exactly like the RGB images, e.g., 171388...png
    # But wait! generate_depth.py saves them as `f"{i:06d}.png"`!
    depth_files = sorted(f for f in os.listdir(DEPTH_DIR) if f.endswith(".png"))
    total_scans = min(len(scan_files), len(depth_files))
    print(f"Found {total_scans} aligned frames")

    pose_ts, pos, quat = load_poses(POSE_CSV)
    interp_pose = build_pose_interpolator(pose_ts, pos, quat)
    imu_t0 = get_imu_t0(IMU_CSV)

    start_idx = args.start
    end_idx = total_scans if args.num_frames is None else min(start_idx + args.num_frames, total_scans)

    for i in range(start_idx, end_idx):
        t_ref = imu_t0 + i / LIDAR_HZ
        T_world_ref = interp_pose(t_ref)
        T_ref_world = np.linalg.inv(T_world_ref)

        all_pts_ego = []
        lo = max(0, i - ACCUM_HALF_WINDOW)
        hi = min(total_scans - 1, i + ACCUM_HALF_WINDOW)

        # 1. LiDAR points
        for j in range(lo, hi + 1):
            xyz_sensor = load_scan(PC_DIR / scan_files[j])
            if len(xyz_sensor) == 0: continue

            if j == i:
                all_pts_ego.append(xyz_sensor)
            else:
                t_j = imu_t0 + j / LIDAR_HZ
                T_world_j = interp_pose(t_j)
                T_ref_j = T_ref_world @ T_world_j
                ones = np.ones((len(xyz_sensor), 1))
                xyz_ref = (T_ref_j @ np.hstack([xyz_sensor, ones]).T).T[:, :3]
                all_pts_ego.append(xyz_ref)
                
        # 2. Dense Depth points (only the reference frame)
        depth_pts = load_depth_points(DEPTH_DIR / depth_files[i])
        if len(depth_pts) > 0:
            all_pts_ego.append(depth_pts)

        if not all_pts_ego:
            continue

        # Load semantics and features if provided
        sem_img = None
        if args.sem_dir:
            sem_path = Path(args.sem_dir) / f"{i:06d}.png"
            if sem_path.exists():
                sem_img = cv2.imread(str(sem_path), cv2.IMREAD_UNCHANGED)
                
        feat_map = None
        if args.feat_dir:
            feat_path = Path(args.feat_dir) / f"{i:06d}.npy"
            if feat_path.exists():
                feat_map = np.load(str(feat_path))

        pts_ego = np.vstack(all_pts_ego)
        bev_height, bev_slope, bev_valid, bev_sem, bev_feat = points_to_bev(pts_ego, sem_img, feat_map)

        out_path = OUTPUT_DIR / f"{i:06d}.npz"
        save_dict = {
            'bev_height': bev_height,
            'bev_slope': bev_slope,
            'bev_valid': bev_valid,
            'bev_fov_mask': FOV_MASK
        }
        if bev_sem is not None: save_dict['bev_semantics'] = bev_sem
        if bev_feat is not None: save_dict['bev_features'] = bev_feat
        
        np.savez_compressed(str(out_path), **save_dict)

        valid_pct = 100 * bev_valid.sum() / FOV_MASK.sum() if FOV_MASK.sum() > 0 else 0
        print(f"  [{i:06d}] {len(pts_ego):>7,} pts (LiDAR+Depth) -> "
              f"{bev_valid.sum():>5}/{FOV_MASK.sum()} valid ({valid_pct:.1f}%)")

        if args.visualize:
            # If processing many frames, only save a few. If < 10, save all.
            step = max(1, (end_idx - start_idx) // 5)
            if (i - start_idx) % step == 0:
                _save_viz(i, bev_height, bev_slope, bev_valid)

def _save_viz(idx, height, slope, valid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    h_vis = np.where(valid, height, np.nan)
    im0 = axes[0].imshow(h_vis, cmap="terrain", aspect="equal")
    axes[0].set_title("Height (m)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    s_vis = np.where(valid, slope, np.nan)
    im1 = axes[1].imshow(s_vis, cmap="magma", vmin=0, vmax=30, aspect="equal")
    axes[1].set_title("Slope (deg)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    axes[2].imshow(FOV_MASK.astype(float) * 0.3 + valid.astype(float) * 0.7, cmap="gray", vmin=0, vmax=1, aspect="equal")
    axes[2].set_title(f"Valid ({valid.sum()}/{FOV_MASK.sum()})")

    for ax in axes:
        ax.set_xlabel("Lateral (Y)")
        ax.set_ylabel("Forward (X)")

    fig.suptitle(f"Hybrid LiDAR+Depth BEV - Frame {idx:06d}", fontsize=14)
    plt.tight_layout()
    viz_path = OUTPUT_DIR / "viz" / f"{idx:06d}.png"
    plt.savefig(str(viz_path), dpi=120)
    plt.close(fig)

if __name__ == "__main__":
    main()
