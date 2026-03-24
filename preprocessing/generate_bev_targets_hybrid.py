"""
Generate BEV ground-truth maps from dense calibrated depth (camera-derived).

Pipeline:
  1. Load dense 16-bit metric depth map (LiDAR-calibrated by generate_depth.py).
  2. Unproject to 3D ego-frame points; filter ledge artifact at camera bottom.
  3. Apply gravity alignment (remove vehicle roll/pitch via pose quaternion).
  4. Filter dynamic objects (vehicle, person) when a semantic image is provided.
  5. Bin into BEV grid (50x50, 25m forward x 25m lateral):
       - Ground layer: points within GROUND_LAYER_THICKNESS above the robust cell floor.
       - Ground height: 95th-percentile of ground-layer points.
       - Non-ground points:
           clearance = min(non_ground_z) - ground_h
           if clearance < HANGING_MIN_CLEARANCE:   ground-based obstacle
               bev_height = max(all_z)             (obstacle top merged into height)
           else:                                    hanging object or clear sky
               bev_height = ground_h
           bev_clearance = clearance  (0 if no non-ground points)
       - Slope: SVD plane fit on ground-layer points in 3x3 cell neighbourhood.
  6. Normalise bev_height: subtract per-frame median raw ground height
     so that 0 = local flat reference plane.
  7. IDW infill for empty cells within INFILL_RADIUS.
  8. Apply Camera FOV mask.
  9. Save .npz: bev_height, bev_clearance, bev_slope, bev_valid,
                bev_confidence, bev_fov_mask [, bev_semantics] [, bev_features]

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

# Depth point filtering
MIN_RANGE_DEPTH   = 1.0    # m
MIN_FORWARD_DEPTH = 3.5    # m — removes 'ledge' artifact at camera bottom
MAX_RANGE_DEPTH   = 50.0   # m

Z_BAND_MIN = -3.0
Z_BAND_MAX =  5.0

# BEV cell thresholds
MIN_PTS_PER_CELL        = 3
GROUND_LAYER_THICKNESS  = 0.40  # m above cell floor → ground layer
GROUND_FLOOR_PERCENTILE = 5     # robust floor: ignore lowest 5% (noise/reflections)
MIN_GROUND_PTS          = 3     # min ground-layer points for a valid cell
GROUND_HEIGHT_PERCENTILE = 95   # percentile within ground layer → height
HANGING_MIN_CLEARANCE   = 0.30  # m — non-ground points below this are ground-based obstacles
MIN_PLANE_PTS           = 6     # min pts in 3x3 neighbourhood for SVD plane fit
INFILL_RADIUS           = 7     # cells
IDW_POWER               = 2     # inverse-distance weighting exponent

# Dynamic object filtering (filtered when sem_img is provided)
DYNAMIC_CLASSES = {8, 9}        # 8 = vehicle, 9 = person

LIDAR_HZ = 10.0

# -- Camera intrinsics & FOV constraints --------------------------------------
FX, FY = 897.929, 974.238
CX, CY = 648.650, 256.277
IMG_H, IMG_W = 720, 1280
PIXEL_STRIDE = 1

# LiDAR -> Camera extrinsic (used for projection and depth unprojection)
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
            x_l = X_MIN + (BEV_RES - 1 - r + 0.5) * CELL_X
            y_l = Y_MIN + (c + 0.5) * CELL_Y
            pt  = T_cam_lidar @ np.array([x_l, y_l, 0.0, 1.0])
            if pt[2] <= 0.1:
                continue
            u = pt[0] * FX / pt[2] + CX
            v = pt[1] * FY / pt[2] + CY
            if 0 <= u < IMG_W and 0 <= v < IMG_H:
                mask[r, c] = True
    return mask


FOV_MASK = build_fov_mask()


# -- Semantic colour palette --------------------------------------------------
SEM_CLASSES = [
    "dirt road", "grass", "tree", "bush", "rock",
    "mud", "water", "sky", "vehicle", "person",
    "building", "fence", "terrain",
]
SEM_COLORS = np.array([
    [0.55, 0.40, 0.20],
    [0.40, 0.75, 0.30],
    [0.13, 0.45, 0.13],
    [0.20, 0.60, 0.20],
    [0.65, 0.60, 0.55],
    [0.50, 0.35, 0.15],
    [0.20, 0.50, 0.90],
    [0.55, 0.75, 0.95],
    [0.95, 0.20, 0.20],
    [0.95, 0.55, 0.10],
    [0.70, 0.70, 0.70],
    [0.85, 0.85, 0.20],
    [0.75, 0.65, 0.45],
], dtype=np.float32)


def _sem_to_rgb(sem, valid):
    rgb = np.ones((BEV_RES, BEV_RES, 3), dtype=np.float32) * 0.15
    for cls_id, colour in enumerate(SEM_COLORS):
        rgb[valid & (sem == cls_id)] = colour
    return rgb


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
        t_c = np.clip(t, ts_u[0], ts_u[-1])
        T   = np.eye(4)
        T[:3, :3] = slerp(t_c).as_matrix()
        T[:3,  3] = [np.interp(t_c, ts_u, pos_u[:, k]) for k in range(3)]
        return T
    return interp_pose


# -- Gravity alignment --------------------------------------------------------
def gravity_align_rotation(T_world_ego):
    """
    Minimal rotation (Rodrigues) that maps the gravity vector in ego frame to [0,0,1].
    Removes vehicle roll/pitch; yaw is preserved.
    """
    g = T_world_ego[:3, :3].T @ np.array([0., 0., 1.])
    g /= np.linalg.norm(g)
    z  = np.array([0., 0., 1.])
    v  = np.cross(g, z)
    s  = np.linalg.norm(v)
    c  = np.dot(g, z)
    if s < 1e-6:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 - c) / (s * s)


# -- Depth loading ------------------------------------------------------------
def load_depth_points(depth_path):
    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        return np.zeros((0, 3))
    depth_sub   = depth_img[::PIXEL_STRIDE, ::PIXEL_STRIDE].astype(np.float32)
    depth_m     = depth_sub / 1000.0
    valid       = (depth_m > MIN_RANGE_DEPTH) & (depth_m < MAX_RANGE_DEPTH)
    Z_c         = depth_m[valid]
    pts_cam     = np.stack([_ray_x[valid] * Z_c, _ray_y[valid] * Z_c, Z_c], axis=1)
    ones        = np.ones((len(pts_cam), 1))
    pts_ego     = (T_lidar_cam @ np.hstack([pts_cam, ones]).T).T[:, :3]
    return pts_ego[pts_ego[:, 0] >= MIN_FORWARD_DEPTH]


# -- BEV helpers --------------------------------------------------------------
def _project_to_camera(pts_ego):
    ones    = np.ones((len(pts_ego), 1))
    pts_cam = (T_cam_lidar @ np.hstack([pts_ego, ones]).T).T
    front   = pts_cam[:, 2] > 0.1
    u = np.full(len(pts_ego), -1, dtype=np.int32)
    v = np.full(len(pts_ego), -1, dtype=np.int32)
    vld = np.zeros(len(pts_ego), dtype=bool)
    if front.any():
        zc     = pts_cam[front, 2]
        uc     = (pts_cam[front, 0] * FX / zc + CX).astype(np.int32)
        vc     = (pts_cam[front, 1] * FY / zc + CY).astype(np.int32)
        in_img = (uc >= 0) & (uc < IMG_W) & (vc >= 0) & (vc < IMG_H)
        idx    = np.where(front)[0][in_img]
        u[idx], v[idx], vld[idx] = uc[in_img], vc[in_img], True
    return u, v, vld


def infill_nearest(grid, valid, radius):
    """Nearest-neighbour infill — for categorical grids (semantics)."""
    if valid.all() or not valid.any():
        return grid.copy()
    dist, idx = ndimage.distance_transform_edt(~valid, return_indices=True)
    out = grid.copy()
    fill = (~valid) & (dist <= radius)
    out[fill] = grid[idx[0][fill], idx[1][fill]]
    return out


def _idw_weights(valid, radius, power):
    den = np.zeros((BEV_RES, BEV_RES), dtype=np.float64)
    dist = ndimage.distance_transform_edt(~valid)
    fill = (~valid) & (dist <= radius)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            d = np.sqrt(dr * dr + dc * dc)
            if d < 1e-6 or d > radius:
                continue
            w    = 1.0 / (d ** power)
            s_r0 = max(0, -dr); s_r1 = min(BEV_RES, BEV_RES - dr)
            s_c0 = max(0, -dc); s_c1 = min(BEV_RES, BEV_RES - dc)
            d_r0 = s_r0 + dr;   d_r1 = s_r1 + dr
            d_c0 = s_c0 + dc;   d_c1 = s_c1 + dc
            den[d_r0:d_r1, d_c0:d_c1] += w * valid[s_r0:s_r1, s_c0:s_c1].astype(np.float64)
    return fill, den


def infill_idw(grid, valid, radius, power=IDW_POWER):
    """IDW infill for scalar (H, W) grids."""
    if valid.all() or not valid.any():
        return grid.copy()
    fill, den = _idw_weights(valid, radius, power)
    if not fill.any():
        return grid.copy()
    num = np.zeros((BEV_RES, BEV_RES), dtype=np.float64)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            d = np.sqrt(dr * dr + dc * dc)
            if d < 1e-6 or d > radius:
                continue
            w    = 1.0 / (d ** power)
            s_r0 = max(0, -dr); s_r1 = min(BEV_RES, BEV_RES - dr)
            s_c0 = max(0, -dc); s_c1 = min(BEV_RES, BEV_RES - dc)
            d_r0 = s_r0 + dr;   d_r1 = s_r1 + dr
            d_c0 = s_c0 + dc;   d_c1 = s_c1 + dc
            sv = valid[s_r0:s_r1, s_c0:s_c1].astype(np.float64)
            sg = grid [s_r0:s_r1, s_c0:s_c1].astype(np.float64)
            num[d_r0:d_r1, d_c0:d_c1] += w * sv * sg
    out = grid.copy()
    has = fill & (den > 0)
    out[has] = (num[has] / den[has]).astype(grid.dtype)
    return out


def infill_idw_nd(grid, valid, radius, power=IDW_POWER):
    """IDW infill for (H, W, D) feature grids — weights computed once."""
    if valid.all() or not valid.any():
        return grid.copy()
    fill, den = _idw_weights(valid, radius, power)
    if not fill.any():
        return grid.copy()
    D   = grid.shape[2]
    num = np.zeros((BEV_RES, BEV_RES, D), dtype=np.float64)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            d = np.sqrt(dr * dr + dc * dc)
            if d < 1e-6 or d > radius:
                continue
            w    = 1.0 / (d ** power)
            s_r0 = max(0, -dr); s_r1 = min(BEV_RES, BEV_RES - dr)
            s_c0 = max(0, -dc); s_c1 = min(BEV_RES, BEV_RES - dc)
            d_r0 = s_r0 + dr;   d_r1 = s_r1 + dr
            d_c0 = s_c0 + dc;   d_c1 = s_c1 + dc
            sv = valid[s_r0:s_r1, s_c0:s_c1].astype(np.float64)
            sg = grid [s_r0:s_r1, s_c0:s_c1].astype(np.float64)
            num[d_r0:d_r1, d_c0:d_c1] += w * sv[:, :, None] * sg
    out = grid.copy()
    has = fill & (den > 0)
    out[has] = (num[has] / den[has][:, None]).astype(grid.dtype)
    return out


def compute_slope_plane_fit(cell_ground_pts, valid):
    """SVD plane fit on ground-layer 3D points pooled from 3x3 neighbourhood."""
    slope = np.zeros((BEV_RES, BEV_RES), dtype=np.float32)
    for r in range(BEV_RES):
        for c in range(BEV_RES):
            if not valid[r, c]:
                continue
            nbr = [cell_ground_pts[k]
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                   if (k := (r + dr, c + dc)) in cell_ground_pts]
            if not nbr:
                continue
            pts = np.vstack(nbr)
            if len(pts) < MIN_PLANE_PTS:
                continue
            centroid = pts.mean(axis=0)
            _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
            normal = Vt[-1]
            if normal[2] < 0:
                normal = -normal
            nz = float(np.clip(np.abs(normal[2]) / np.linalg.norm(normal), 0.0, 1.0))
            slope[r, c] = float(np.degrees(np.arccos(nz)))
    return slope


# -- Main BEV construction ----------------------------------------------------
def points_to_bev(pts_ego, sem_img=None, feat_map=None):
    """
    Bin gravity-aligned depth points into the BEV grid.

    bev_height:    ground surface height, OR obstacle top when a ground-based
                   obstacle is present (clearance < HANGING_MIN_CLEARANCE).
                   Normalised so that median ground = 0.
    bev_clearance: gap (m) between ground surface and the lowest non-ground point.
                   0 when no non-ground points or ground-based obstacle.
                   > 0 when a hanging object is present (passable underneath).
    bev_slope:     terrain slope in degrees (always from ground layer only).
    bev_valid:     bool mask — True where enough ground points were found.
    bev_confidence: number of ground-layer points per cell.
    """
    x, y, z = pts_ego[:, 0], pts_ego[:, 1], pts_ego[:, 2]

    # -- 1. Dynamic object filtering ------------------------------------------
    u_cam = v_cam = valid_proj = None
    if sem_img is not None:
        u_cam, v_cam, valid_proj = _project_to_camera(pts_ego)
        dyn = np.zeros(len(pts_ego), dtype=bool)
        if valid_proj.any():
            dyn[valid_proj] = np.isin(sem_img[v_cam[valid_proj], u_cam[valid_proj]],
                                      list(DYNAMIC_CLASSES))
        keep = ~dyn
        pts_ego, x, y, z = pts_ego[keep], x[keep], y[keep], z[keep]
        u_cam, v_cam, valid_proj = _project_to_camera(pts_ego)
    elif feat_map is not None:
        u_cam, v_cam, valid_proj = _project_to_camera(pts_ego)

    # -- 2. Spatial bounds filter ---------------------------------------------
    bounds = ((x >= X_MIN) & (x < X_MAX) &
              (y >= Y_MIN) & (y < Y_MAX) &
              (z >= Z_BAND_MIN) & (z < Z_BAND_MAX))
    x, y, z = x[bounds], y[bounds], z[bounds]
    pts_f   = pts_ego[bounds]
    if valid_proj is not None:
        u_f, v_f, vp_f = u_cam[bounds], v_cam[bounds], valid_proj[bounds]
    else:
        u_f = v_f = vp_f = None

    bev_sem  = np.full((BEV_RES, BEV_RES), -1, dtype=np.int32) if sem_img  is not None else None
    bev_feat = np.zeros((BEV_RES, BEV_RES, feat_map.shape[-1]), dtype=np.float32) if feat_map is not None else None

    _empty = (np.zeros((BEV_RES, BEV_RES), dtype=np.float32),   # height
              np.zeros((BEV_RES, BEV_RES), dtype=np.float32),   # clearance
              np.zeros((BEV_RES, BEV_RES), dtype=np.float32),   # slope
              np.zeros((BEV_RES, BEV_RES), dtype=bool),          # valid
              np.zeros((BEV_RES, BEV_RES), dtype=np.uint16),     # confidence
              bev_sem, bev_feat)
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

    bev_height     = np.full((BEV_RES, BEV_RES), np.nan, dtype=np.float32)
    bev_ground_raw = np.full((BEV_RES, BEV_RES), np.nan, dtype=np.float32)  # for normalisation
    bev_clearance  = np.zeros((BEV_RES, BEV_RES), dtype=np.float32)
    bev_valid      = np.zeros((BEV_RES, BEV_RES), dtype=bool)
    bev_confidence = np.zeros((BEV_RES, BEV_RES), dtype=np.uint16)

    cell_idx    = row * BEV_RES + col
    order       = np.argsort(cell_idx)
    cell_idx_s  = cell_idx[order]
    z_s         = z[order]
    x_s         = pts_f[order, 0]
    y_s         = pts_f[order, 1]

    boundaries  = np.searchsorted(cell_idx_s, np.arange(BEV_RES * BEV_RES))
    boundaries  = np.append(boundaries, len(cell_idx_s))

    pt_sem_s  = pt_sem[order]  if sem_img  is not None else None
    pt_feat_s = pt_feat[order] if feat_map is not None else None
    vp_s      = vp_f[order]    if vp_f     is not None else None

    cell_ground_pts: dict = {}

    # -- 5. Per-cell loop -----------------------------------------------------
    for cell in range(BEV_RES * BEV_RES):
        start, end = boundaries[cell], boundaries[cell + 1]
        if end - start < MIN_PTS_PER_CELL:
            continue

        cell_z = z_s[start:end]

        cell_floor = np.percentile(cell_z, GROUND_FLOOR_PERCENTILE)
        ground_m   = cell_z <= (cell_floor + GROUND_LAYER_THICKNESS)
        ground_z   = cell_z[ground_m]

        if len(ground_z) < MIN_GROUND_PTS:
            continue

        r_idx = cell // BEV_RES
        c_idx = cell  % BEV_RES

        ground_h = float(np.percentile(ground_z, GROUND_HEIGHT_PERCENTILE))
        bev_ground_raw[r_idx, c_idx] = ground_h

        # Non-ground points
        non_ground_z = cell_z[~ground_m]
        if len(non_ground_z) > 0:
            clearance = float(np.min(non_ground_z)) - ground_h
            if clearance < HANGING_MIN_CLEARANCE:
                # Ground-based obstacle: merge top into height map
                bev_height[r_idx, c_idx]    = float(np.max(non_ground_z))
                bev_clearance[r_idx, c_idx] = 0.0
            else:
                # Hanging object: height stays at ground, record clearance
                bev_height[r_idx, c_idx]    = ground_h
                bev_clearance[r_idx, c_idx] = clearance
        else:
            bev_height[r_idx, c_idx]    = ground_h
            bev_clearance[r_idx, c_idx] = 0.0

        bev_valid[r_idx, c_idx]      = True
        bev_confidence[r_idx, c_idx] = min(len(ground_z), 65535)

        # 3D ground points for plane fit
        gi = np.where(ground_m)[0]
        cell_ground_pts[(r_idx, c_idx)] = np.column_stack(
            [x_s[start:end][gi], y_s[start:end][gi], ground_z])

        if sem_img is not None:
            cs = pt_sem_s[start:end]
            cs_v = cs[cs >= 0]
            if len(cs_v) > 0:
                vals, counts = np.unique(cs_v, return_counts=True)
                bev_sem[r_idx, c_idx] = vals[np.argmax(counts)]

        if feat_map is not None:
            cf, cv = pt_feat_s[start:end], vp_s[start:end]
            if cv.any():
                bev_feat[r_idx, c_idx] = np.mean(cf[cv], axis=0)

    # -- 6. Height normalisation: subtract per-frame median ground height -----
    ground_ref = float(np.nanmedian(bev_ground_raw[bev_valid])) if bev_valid.any() else 0.0
    bev_height[bev_valid] -= ground_ref

    # -- 7. Slope via 3D plane fit --------------------------------------------
    bev_slope = compute_slope_plane_fit(cell_ground_pts, bev_valid)

    # -- 8. IDW infill --------------------------------------------------------
    bev_height_f    = infill_idw(np.where(bev_valid, bev_height, 0.0).astype(np.float32),    bev_valid, INFILL_RADIUS)
    bev_clearance_f = infill_idw(bev_clearance, bev_valid, INFILL_RADIUS)
    bev_slope_f     = infill_idw(bev_slope,     bev_valid, INFILL_RADIUS)

    valid_f = bev_valid | (infill_idw(bev_valid.astype(np.float32), bev_valid, INFILL_RADIUS) > 0)

    bev_height_f[~valid_f]    = 0.0
    bev_clearance_f[~valid_f] = 0.0
    bev_slope_f[~valid_f]     = 0.0

    # -- 9. Feature IDW infill ------------------------------------------------
    if bev_feat is not None:
        bev_feat_f = infill_idw_nd(bev_feat, bev_valid, INFILL_RADIUS)
        bev_feat_f[~valid_f]  = 0.0
        bev_feat_f[~FOV_MASK] = 0.0
    else:
        bev_feat_f = None

    # -- 10. Semantic nearest-neighbour infill --------------------------------
    if bev_sem is not None:
        bev_sem_f = infill_nearest(bev_sem, bev_sem >= 0, INFILL_RADIUS)
        bev_sem_f[~valid_f]  = -1
        bev_sem_f[~FOV_MASK] = -1
    else:
        bev_sem_f = None

    # -- 11. FOV mask ---------------------------------------------------------
    bev_height_f[~FOV_MASK]    = 0.0
    bev_clearance_f[~FOV_MASK] = 0.0
    bev_slope_f[~FOV_MASK]     = 0.0
    valid_f &= FOV_MASK

    return bev_height_f, bev_clearance_f, bev_slope_f, valid_f, bev_confidence, bev_sem_f, bev_feat_f


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

    depth_files = sorted(f for f in os.listdir(DEPTH_DIR) if f.endswith(".png"))
    total_frames = len(depth_files)
    print(f"Found {total_frames} depth frames")

    pose_ts, pos, quat = load_poses(POSE_CSV)
    interp_pose = build_pose_interpolator(pose_ts, pos, quat)
    imu_t0 = get_imu_t0(IMU_CSV)

    start_idx = args.start
    end_idx   = total_frames if args.num_frames is None else min(start_idx + args.num_frames, total_frames)

    for i in range(start_idx, end_idx):
        t_ref       = imu_t0 + i / LIDAR_HZ
        T_world_ref = interp_pose(t_ref)
        R_level     = gravity_align_rotation(T_world_ref)

        # Depth points (reference frame only — already LiDAR-calibrated)
        pts_ego = load_depth_points(DEPTH_DIR / depth_files[i])
        if len(pts_ego) == 0:
            continue

        # Gravity alignment
        pts_ego = (R_level @ pts_ego.T).T

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

        bev_height, bev_clearance, bev_slope, bev_valid, bev_confidence, bev_sem, bev_feat = \
            points_to_bev(pts_ego, sem_img, feat_map)

        save_dict = {
            'bev_height':     bev_height,
            'bev_clearance':  bev_clearance,
            'bev_slope':      bev_slope,
            'bev_valid':      bev_valid,
            'bev_confidence': bev_confidence,
            'bev_fov_mask':   FOV_MASK,
        }
        if bev_sem  is not None: save_dict['bev_semantics'] = bev_sem
        if bev_feat is not None: save_dict['bev_features']  = bev_feat

        np.savez_compressed(str(OUTPUT_DIR / f"{i:06d}.npz"), **save_dict)

        valid_pct = 100 * bev_valid.sum() / FOV_MASK.sum() if FOV_MASK.sum() > 0 else 0
        print(f"  [{i:06d}] {len(pts_ego):>7,} depth pts -> "
              f"{bev_valid.sum():>5}/{FOV_MASK.sum()} valid ({valid_pct:.1f}%)")

        if args.visualize:
            step = max(1, (end_idx - start_idx) // 5)
            if (i - start_idx) % step == 0:
                _save_viz(i, bev_height, bev_clearance, bev_slope, bev_valid, bev_confidence, bev_sem)


def _save_viz(idx, height, clearance, slope, valid, confidence, sem=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n_panels = 6 if sem is not None else 5
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    im0 = axes[0].imshow(np.where(valid, height, np.nan), cmap="coolwarm", aspect="equal")
    axes[0].set_title("Height (m, normalised)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(np.where(valid, clearance, np.nan), cmap="viridis", vmin=0, vmax=3, aspect="equal")
    axes[1].set_title("Clearance (m)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(np.where(valid, slope, np.nan), cmap="magma", vmin=0, vmax=30, aspect="equal")
    axes[2].set_title("Slope (deg)")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    im3 = axes[3].imshow(confidence, cmap="hot", aspect="equal")
    axes[3].set_title("Ground Pt Count")
    plt.colorbar(im3, ax=axes[3], shrink=0.8)

    axes[4].imshow(FOV_MASK.astype(float) * 0.3 + valid.astype(float) * 0.7,
                   cmap="gray", vmin=0, vmax=1, aspect="equal")
    axes[4].set_title(f"Valid ({valid.sum()}/{FOV_MASK.sum()})")

    if sem is not None:
        sem_valid = valid & (sem >= 0)
        axes[5].imshow(_sem_to_rgb(sem, sem_valid), aspect="equal")
        axes[5].set_title("Semantics")
        present = sorted({int(c) for c in sem[sem_valid]} if sem_valid.any() else [])
        legend  = [Patch(facecolor=SEM_COLORS[c], label=SEM_CLASSES[c])
                   for c in present if c < len(SEM_CLASSES)]
        if legend:
            axes[5].legend(handles=legend, loc="lower right", fontsize=6,
                           framealpha=0.7, ncol=2)

    for ax in axes:
        ax.set_xlabel("Lateral (Y)")
        ax.set_ylabel("Forward (X)")

    fig.suptitle(f"BEV Ground Truth — Frame {idx:06d}", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "viz" / f"{idx:06d}.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
