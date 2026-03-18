"""
Generate Hybrid BEV height + slope maps from accumulated LiDAR scans + Dense Relative Depth.

Pipeline:
  1. Accumulate LiDAR scans (e.g. +/-7) into ego frame using poses.csv.
  2. Load dense 16-bit metric depth map (already calibrated to LiDAR in generate_depth.py).
  3. Unproject dense depth to 3D, transform from camera to ego frame.
  4. Filter out dense depth points < 3.5m forward (to remove the 'ledge' artifact).
  5. Tag LiDAR points source=0, depth points source=1.
  6. Apply gravity alignment (remove vehicle roll/pitch via pose quaternion).
  7. Filter dynamic objects (vehicle, person) when a semantic image is provided.
  8. Bin into BEV grid:
       - LiDAR-priority: depth points only used in cells with no LiDAR coverage.
       - Ground layer: points within GROUND_LAYER_THICKNESS above the robust cell floor.
       - Ground height: 95th-percentile of ground-layer points (robust to outliers).
       - Above-ground height: max(all pts) - ground height.
       - Slope: 3D plane fit on ground-layer points in 3x3 cell neighbourhood.
  9. IDW infill for empty cells within INFILL_RADIUS.
  10. Apply Camera FOV mask.
  11. Save .npz: bev_height, bev_above_ground, bev_slope, bev_valid, bev_confidence, bev_fov_mask.

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
MIN_RANGE_LIDAR   = 1.0    # m
MAX_RANGE_LIDAR   = 50.0   # m
MIN_FORWARD_DEPTH = 3.5    # m — removes 'ledge' artifact at camera bottom
MAX_RANGE_DEPTH   = 50.0   # m

Z_BAND_MIN = -3.0
Z_BAND_MAX =  5.0

# BEV cell thresholds
MIN_PTS_PER_CELL       = 3
GROUND_LAYER_THICKNESS = 0.40  # m above cell floor → ground layer
GROUND_FLOOR_PERCENTILE = 5    # robust floor: ignore lowest 5% (noise/reflections)
MIN_GROUND_PTS         = 3     # min ground-layer points for a valid cell
GROUND_HEIGHT_PERCENTILE = 95  # percentile of ground layer → height (robust to outliers)
MIN_PLANE_PTS          = 6     # min pts in 3x3 neighbourhood for SVD plane fit
INFILL_RADIUS          = 7     # cells
IDW_POWER              = 2     # inverse-distance weighting exponent

# Dynamic object filtering (filtered when sem_img is provided)
DYNAMIC_CLASSES = {8, 9}   # 8 = vehicle, 9 = person

# Point source tags
SRC_LIDAR = 0
SRC_DEPTH = 1

LIDAR_HZ = 10.0

# -- Camera intrinsics & FOV constraints --------------------------------------
FX, FY = 897.929, 974.238
CX, CY = 648.650, 256.277
IMG_H, IMG_W = 720, 1280
PIXEL_STRIDE = 1

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
            pt_lidar = np.array([x_lidar, y_lidar, 0.0, 1.0])
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
    ts   = data[:, 0] * 1e-9
    pos  = data[:, 1:4]
    quat = data[:, 4:8]
    return ts, pos, quat


def get_imu_t0(imu_csv):
    with open(imu_csv, "r") as f:
        f.readline()
        return float(f.readline().split(",")[0])


def build_pose_interpolator(pose_ts, pos, quat):
    _, unique_idx = np.unique(pose_ts, return_index=True)
    unique_idx = np.sort(unique_idx)
    ts_u   = pose_ts[unique_idx]
    pos_u  = pos[unique_idx]
    quat_u = quat[unique_idx]
    rots   = Rotation.from_quat(quat_u)
    slerp  = Slerp(ts_u, rots)

    def interp_pose(t):
        t_clamped = np.clip(t, ts_u[0], ts_u[-1])
        px  = np.interp(t_clamped, ts_u, pos_u[:, 0])
        py  = np.interp(t_clamped, ts_u, pos_u[:, 1])
        pz  = np.interp(t_clamped, ts_u, pos_u[:, 2])
        rot = slerp(t_clamped)
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3]  = [px, py, pz]
        return T
    return interp_pose


# -- Gravity alignment --------------------------------------------------------
def gravity_align_rotation(T_world_ego):
    """
    Return a 3x3 rotation R_level that removes vehicle roll/pitch.
    R_level @ v_ego gives v in a gravity-aligned frame (Z = world-up).
    Yaw direction (horizontal forward) is preserved.
    Uses the Rodrigues formula to build the minimal rotation from the
    gravity vector (in ego frame) to [0,0,1].
    """
    R = T_world_ego[:3, :3]
    g_ego = R.T @ np.array([0., 0., 1.])   # world-up expressed in ego frame
    g_ego /= np.linalg.norm(g_ego)
    z = np.array([0., 0., 1.])
    v = np.cross(g_ego, z)
    s = np.linalg.norm(v)
    c = np.dot(g_ego, z)
    if s < 1e-6:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 - c) / (s * s)


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
    depth_m   = depth_sub / 1000.0
    valid_depth = (depth_m > MIN_RANGE_LIDAR) & (depth_m < MAX_RANGE_DEPTH)
    Z_c = depth_m[valid_depth]
    X_c = _ray_x[valid_depth] * Z_c
    Y_c = _ray_y[valid_depth] * Z_c
    pts_cam  = np.stack([X_c, Y_c, Z_c], axis=1)
    ones     = np.ones((len(pts_cam), 1))
    pts_ego  = (T_lidar_cam @ np.hstack([pts_cam, ones]).T).T[:, :3]
    return pts_ego[pts_ego[:, 0] >= MIN_FORWARD_DEPTH]


# -- BEV helpers --------------------------------------------------------------
def _project_to_camera(pts_ego):
    """Project ego-frame points into the reference camera. Returns (u, v, valid_mask)."""
    ones    = np.ones((len(pts_ego), 1))
    pts_cam = (T_cam_lidar @ np.hstack([pts_ego, ones]).T).T
    front   = pts_cam[:, 2] > 0.1
    u = np.full(len(pts_ego), -1, dtype=np.int32)
    v = np.full(len(pts_ego), -1, dtype=np.int32)
    valid = np.zeros(len(pts_ego), dtype=bool)
    if front.any():
        zc  = pts_cam[front, 2]
        uc  = (pts_cam[front, 0] * FX / zc + CX).astype(np.int32)
        vc  = (pts_cam[front, 1] * FY / zc + CY).astype(np.int32)
        in_img = (uc >= 0) & (uc < IMG_W) & (vc >= 0) & (vc < IMG_H)
        idx = np.where(front)[0][in_img]
        u[idx] = uc[in_img]
        v[idx] = vc[in_img]
        valid[idx] = True
    return u, v, valid


def infill_idw(grid, valid, radius, power=IDW_POWER):
    """
    Inverse-distance weighted infill for invalid cells within `radius` cells of
    a valid neighbour.  Vectorised over offsets — O(radius^2) NumPy passes.
    """
    if valid.all() or not valid.any():
        return grid.copy()
    dist    = ndimage.distance_transform_edt(~valid)
    fill    = (~valid) & (dist <= radius)
    if not fill.any():
        return grid.copy()
    num = np.zeros((BEV_RES, BEV_RES), dtype=np.float64)
    den = np.zeros((BEV_RES, BEV_RES), dtype=np.float64)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            d = np.sqrt(dr * dr + dc * dc)
            if d < 1e-6 or d > radius:
                continue
            w   = 1.0 / (d ** power)
            # Source slice [s_r0:s_r1, s_c0:s_c1] -> dest slice offset by (dr, dc)
            s_r0 = max(0, -dr); s_r1 = min(BEV_RES, BEV_RES - dr)
            s_c0 = max(0, -dc); s_c1 = min(BEV_RES, BEV_RES - dc)
            d_r0 = s_r0 + dr;   d_r1 = s_r1 + dr
            d_c0 = s_c0 + dc;   d_c1 = s_c1 + dc
            sv = valid[s_r0:s_r1, s_c0:s_c1].astype(np.float64)
            sg = grid [s_r0:s_r1, s_c0:s_c1].astype(np.float64)
            num[d_r0:d_r1, d_c0:d_c1] += w * sv * sg
            den[d_r0:d_r1, d_c0:d_c1] += w * sv
    out = grid.copy()
    has = fill & (den > 0)
    out[has] = (num[has] / den[has]).astype(grid.dtype)
    return out


def compute_slope_plane_fit(cell_ground_pts, valid):
    """
    Per-cell slope via SVD plane fit on ground-layer 3D points pooled from
    the 3x3 cell neighbourhood.  Returns slope in degrees from vertical.
    Falls back to 0° when fewer than MIN_PLANE_PTS points are available.
    """
    slope = np.zeros((BEV_RES, BEV_RES), dtype=np.float32)
    for r in range(BEV_RES):
        for c in range(BEV_RES):
            if not valid[r, c]:
                continue
            nbr = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    key = (r + dr, c + dc)
                    if key in cell_ground_pts:
                        nbr.append(cell_ground_pts[key])
            if not nbr:
                continue
            pts = np.vstack(nbr)
            if len(pts) < MIN_PLANE_PTS:
                continue
            centroid = pts.mean(axis=0)
            _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
            normal = Vt[-1]          # smallest singular value → plane normal
            if normal[2] < 0:
                normal = -normal
            nz = float(np.clip(np.abs(normal[2]) / np.linalg.norm(normal), 0.0, 1.0))
            slope[r, c] = float(np.degrees(np.arccos(nz)))
    return slope


# -- Main BEV construction ----------------------------------------------------
def points_to_bev(pts_ego, source=None, sem_img=None, feat_map=None):
    """
    Bin gravity-aligned ego-frame points into the BEV grid.

    Args:
        pts_ego:  (N, 3) float64
        source:   (N,)  int8  — SRC_LIDAR=0, SRC_DEPTH=1
        sem_img:  optional (H, W) uint8 semantic label image
        feat_map: optional (H, W, D) float32 feature map

    Returns:
        bev_height, bev_above_ground, bev_slope, bev_valid,
        bev_confidence, bev_sem, bev_feat
    """
    if source is None:
        source = np.zeros(len(pts_ego), dtype=np.int8)

    x, y, z = pts_ego[:, 0], pts_ego[:, 1], pts_ego[:, 2]

    # -- 1. Dynamic object filtering (requires sem_img) -----------------------
    u_cam = v_cam = valid_proj = None
    if sem_img is not None:
        u_cam, v_cam, valid_proj = _project_to_camera(pts_ego)
        dyn = np.zeros(len(pts_ego), dtype=bool)
        if valid_proj.any():
            sem_vals = sem_img[v_cam[valid_proj], u_cam[valid_proj]]
            dyn[valid_proj] = np.isin(sem_vals, list(DYNAMIC_CLASSES))
        keep = ~dyn
        pts_ego, x, y, z, source = (
            pts_ego[keep], x[keep], y[keep], z[keep], source[keep]
        )
        # Re-project after removing dynamic points
        u_cam, v_cam, valid_proj = _project_to_camera(pts_ego)
    elif feat_map is not None:
        u_cam, v_cam, valid_proj = _project_to_camera(pts_ego)

    # -- 2. Spatial bounds filter ---------------------------------------------
    bounds = (
        (x >= X_MIN) & (x < X_MAX) &
        (y >= Y_MIN) & (y < Y_MAX) &
        (z >= Z_BAND_MIN) & (z < Z_BAND_MAX)
    )
    x, y, z, source = x[bounds], y[bounds], z[bounds], source[bounds]
    pts_f = pts_ego[bounds]
    if valid_proj is not None:
        u_f, v_f, vp_f = u_cam[bounds], v_cam[bounds], valid_proj[bounds]
    else:
        u_f = v_f = vp_f = None

    bev_sem  = np.full((BEV_RES, BEV_RES), -1, dtype=np.int32) if sem_img  is not None else None
    bev_feat = np.zeros((BEV_RES, BEV_RES, feat_map.shape[-1]), dtype=np.float32) if feat_map is not None else None

    _empty = (
        np.zeros((BEV_RES, BEV_RES), dtype=np.float32),
        np.zeros((BEV_RES, BEV_RES), dtype=np.float32),
        np.zeros((BEV_RES, BEV_RES), dtype=np.float32),
        np.zeros((BEV_RES, BEV_RES), dtype=bool),
        np.zeros((BEV_RES, BEV_RES), dtype=np.uint16),
        bev_sem, bev_feat,
    )
    if len(x) == 0:
        return _empty

    # -- 3. Per-point semantic / feature lookup -------------------------------
    pt_sem = np.full(len(x), -1, dtype=np.int32)
    pt_feat = None
    if sem_img is not None and vp_f is not None and vp_f.any():
        pt_sem[vp_f] = sem_img[v_f[vp_f], u_f[vp_f]]
    if feat_map is not None and vp_f is not None and vp_f.any():
        pt_feat = np.zeros((len(x), feat_map.shape[-1]), dtype=np.float32)
        pt_feat[vp_f] = feat_map[v_f[vp_f], u_f[vp_f]]

    # -- 4. BEV cell indices --------------------------------------------------
    col = BEV_RES - 1 - np.floor((y - Y_MIN) / CELL_Y).astype(int)
    row = BEV_RES - 1 - np.floor((x - X_MIN) / CELL_X).astype(int)
    col = np.clip(col, 0, BEV_RES - 1)
    row = np.clip(row, 0, BEV_RES - 1)

    bev_height       = np.full((BEV_RES, BEV_RES), np.nan, dtype=np.float32)
    bev_above_ground = np.zeros((BEV_RES, BEV_RES), dtype=np.float32)
    bev_valid        = np.zeros((BEV_RES, BEV_RES), dtype=bool)
    bev_confidence   = np.zeros((BEV_RES, BEV_RES), dtype=np.uint16)

    cell_idx         = row * BEV_RES + col
    order            = np.argsort(cell_idx)
    cell_idx_s       = cell_idx[order]
    z_s              = z[order]
    x_s              = pts_f[order, 0]
    y_s              = pts_f[order, 1]
    src_s            = source[order]

    boundaries = np.searchsorted(cell_idx_s, np.arange(BEV_RES * BEV_RES))
    boundaries = np.append(boundaries, len(cell_idx_s))

    pt_sem_s  = pt_sem[order]  if sem_img  is not None else None
    pt_feat_s = pt_feat[order] if feat_map is not None else None
    vp_s      = vp_f[order]    if vp_f     is not None else None

    # (r, c) -> (N, 3) array of ground-layer 3D points for plane fitting
    cell_ground_pts: dict = {}

    # -- 5. Per-cell loop -----------------------------------------------------
    for cell in range(BEV_RES * BEV_RES):
        start, end = boundaries[cell], boundaries[cell + 1]
        if end - start < MIN_PTS_PER_CELL:
            continue

        cell_z   = z_s[start:end]
        cell_src = src_s[start:end]

        # LiDAR-priority: prefer LiDAR points; fall back to depth only in gaps
        lidar_m = cell_src == SRC_LIDAR
        pool_z  = cell_z[lidar_m] if lidar_m.any() else cell_z

        if len(pool_z) < MIN_PTS_PER_CELL:
            continue

        cell_floor  = np.percentile(pool_z, GROUND_FLOOR_PERCENTILE)
        ground_m    = pool_z <= (cell_floor + GROUND_LAYER_THICKNESS)
        ground_z    = pool_z[ground_m]

        if len(ground_z) < MIN_GROUND_PTS:
            continue

        r_idx = cell // BEV_RES
        c_idx = cell  % BEV_RES

        ground_h = float(np.percentile(ground_z, GROUND_HEIGHT_PERCENTILE))
        bev_height[r_idx, c_idx]       = ground_h
        bev_above_ground[r_idx, c_idx] = float(np.max(cell_z)) - ground_h
        bev_valid[r_idx, c_idx]        = True
        bev_confidence[r_idx, c_idx]   = min(len(ground_z), 65535)

        # Store 3D ground-layer coords for plane fit (LiDAR-priority pool)
        if lidar_m.any():
            px = x_s[start:end][lidar_m]
            py = y_s[start:end][lidar_m]
        else:
            px = x_s[start:end]
            py = y_s[start:end]
        gi = np.where(ground_m)[0]
        cell_ground_pts[(r_idx, c_idx)] = np.column_stack([px[gi], py[gi], ground_z])

        # Semantic majority vote
        if sem_img is not None:
            cs = pt_sem_s[start:end]
            cs_v = cs[cs >= 0]
            if len(cs_v) > 0:
                vals, counts = np.unique(cs_v, return_counts=True)
                bev_sem[r_idx, c_idx] = vals[np.argmax(counts)]

        # Feature mean
        if feat_map is not None:
            cf  = pt_feat_s[start:end]
            cv  = vp_s[start:end]
            if cv.any():
                bev_feat[r_idx, c_idx] = np.mean(cf[cv], axis=0)

    # -- 6. Slope via 3D plane fit --------------------------------------------
    bev_slope = compute_slope_plane_fit(cell_ground_pts, bev_valid)

    # -- 7. IDW infill --------------------------------------------------------
    bev_height_f       = infill_idw(np.where(bev_valid, bev_height, 0.0).astype(np.float32), bev_valid, INFILL_RADIUS)
    bev_above_ground_f = infill_idw(bev_above_ground, bev_valid, INFILL_RADIUS)
    bev_slope_f        = infill_idw(bev_slope,         bev_valid, INFILL_RADIUS)

    valid_f = bev_valid | (infill_idw(bev_valid.astype(np.float32), bev_valid, INFILL_RADIUS) > 0)

    bev_height_f[~valid_f]       = 0.0
    bev_above_ground_f[~valid_f] = 0.0
    bev_slope_f[~valid_f]        = 0.0

    # -- 8. FOV mask ----------------------------------------------------------
    bev_height_f[~FOV_MASK]       = 0.0
    bev_above_ground_f[~FOV_MASK] = 0.0
    bev_slope_f[~FOV_MASK]        = 0.0
    valid_f &= FOV_MASK

    return bev_height_f, bev_above_ground_f, bev_slope_f, valid_f, bev_confidence, bev_sem, bev_feat


# -- Main pipeline ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--visualize",  action="store_true")
    parser.add_argument("--start",      type=int, default=0)
    parser.add_argument("--sem_dir",    type=str, default="")
    parser.add_argument("--feat_dir",   type=str, default="")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        (OUTPUT_DIR / "viz").mkdir(exist_ok=True)

    scan_files  = sorted(f for f in os.listdir(PC_DIR)    if f.endswith(".npy"))
    depth_files = sorted(f for f in os.listdir(DEPTH_DIR) if f.endswith(".png"))
    total_scans = min(len(scan_files), len(depth_files))
    print(f"Found {total_scans} aligned frames")

    pose_ts, pos, quat = load_poses(POSE_CSV)
    interp_pose = build_pose_interpolator(pose_ts, pos, quat)
    imu_t0 = get_imu_t0(IMU_CSV)

    start_idx = args.start
    end_idx   = total_scans if args.num_frames is None else min(start_idx + args.num_frames, total_scans)

    for i in range(start_idx, end_idx):
        t_ref       = imu_t0 + i / LIDAR_HZ
        T_world_ref = interp_pose(t_ref)
        T_ref_world = np.linalg.inv(T_world_ref)

        # Gravity-levelling rotation for this frame
        R_level = gravity_align_rotation(T_world_ref)

        all_pts:  list = []
        all_srcs: list = []
        lo = max(0, i - ACCUM_HALF_WINDOW)
        hi = min(total_scans - 1, i + ACCUM_HALF_WINDOW)

        # 1. Accumulated LiDAR
        for j in range(lo, hi + 1):
            xyz_s = load_scan(PC_DIR / scan_files[j])
            if len(xyz_s) == 0:
                continue
            if j == i:
                all_pts.append(xyz_s)
            else:
                t_j       = imu_t0 + j / LIDAR_HZ
                T_world_j = interp_pose(t_j)
                T_ref_j   = T_ref_world @ T_world_j
                ones      = np.ones((len(xyz_s), 1))
                all_pts.append((T_ref_j @ np.hstack([xyz_s, ones]).T).T[:, :3])
            all_srcs.append(np.full(len(all_pts[-1]), SRC_LIDAR, dtype=np.int8))

        # 2. Dense depth (reference frame only)
        depth_pts = load_depth_points(DEPTH_DIR / depth_files[i])
        if len(depth_pts) > 0:
            all_pts.append(depth_pts)
            all_srcs.append(np.full(len(depth_pts), SRC_DEPTH, dtype=np.int8))

        if not all_pts:
            continue

        pts_ego = np.vstack(all_pts)
        source  = np.concatenate(all_srcs)

        # 3. Gravity alignment — remove vehicle roll/pitch
        pts_ego = (R_level @ pts_ego.T).T

        # 4. Optional semantics / features
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

        bev_height, bev_above_ground, bev_slope, bev_valid, bev_confidence, bev_sem, bev_feat = \
            points_to_bev(pts_ego, source, sem_img, feat_map)

        save_dict = {
            'bev_height':       bev_height,
            'bev_above_ground': bev_above_ground,
            'bev_slope':        bev_slope,
            'bev_valid':        bev_valid,
            'bev_confidence':   bev_confidence,
            'bev_fov_mask':     FOV_MASK,
        }
        if bev_sem  is not None: save_dict['bev_semantics'] = bev_sem
        if bev_feat is not None: save_dict['bev_features']  = bev_feat

        np.savez_compressed(str(OUTPUT_DIR / f"{i:06d}.npz"), **save_dict)

        valid_pct = 100 * bev_valid.sum() / FOV_MASK.sum() if FOV_MASK.sum() > 0 else 0
        print(f"  [{i:06d}] {len(pts_ego):>7,} pts -> "
              f"{bev_valid.sum():>5}/{FOV_MASK.sum()} valid ({valid_pct:.1f}%)")

        if args.visualize:
            step = max(1, (end_idx - start_idx) // 5)
            if (i - start_idx) % step == 0:
                _save_viz(i, bev_height, bev_above_ground, bev_slope, bev_valid, bev_confidence)


def _save_viz(idx, height, above_ground, slope, valid, confidence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    im0 = axes[0].imshow(np.where(valid, height, np.nan), cmap="terrain", aspect="equal")
    axes[0].set_title("Ground Height (m)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(np.where(valid, above_ground, np.nan), cmap="hot", vmin=0, vmax=3, aspect="equal")
    axes[1].set_title("Above Ground (m)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(np.where(valid, slope, np.nan), cmap="magma", vmin=0, vmax=30, aspect="equal")
    axes[2].set_title("Slope (deg)")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    im3 = axes[3].imshow(confidence, cmap="viridis", aspect="equal")
    axes[3].set_title("Ground Pt Count")
    plt.colorbar(im3, ax=axes[3], shrink=0.8)

    axes[4].imshow(
        FOV_MASK.astype(float) * 0.3 + valid.astype(float) * 0.7,
        cmap="gray", vmin=0, vmax=1, aspect="equal"
    )
    axes[4].set_title(f"Valid ({valid.sum()}/{FOV_MASK.sum()})")

    for ax in axes:
        ax.set_xlabel("Lateral (Y)")
        ax.set_ylabel("Forward (X)")

    fig.suptitle(f"Hybrid LiDAR+Depth BEV — Frame {idx:06d}", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "viz" / f"{idx:06d}.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
