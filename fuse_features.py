"""
fuse_features.py
Lift DINOv2 feature embeddings into 3D point clouds using depth backprojection.
Each 3D point carries a high-dimensional feature vector.

Output:
  - .ply files with PCA false-color visualization (3 channels from feature PCA)
  - .npz files with full feature vectors per point for downstream use
"""

import os
import glob
import numpy as np
from PIL import Image

# Import shared pipeline functions
from fuse_outdoors import (
    write_ply, load_intrinsics, load_extrinsics, load_imu_data,
    dead_reckon_positions, get_world_pose_at_timestamp,
    find_closest_matches,
    CAMERAS, IMAGE_DIR_BASE, DEPTH_DIR_BASE, DEFAULT_INTRINSICS
)

# Configuration
HOME = os.path.expanduser("~")
FEATURES_DIR = os.path.join(HOME, "Downloads", "the_great_outdoors_data_features")
OUTPUT_DIR = os.path.join(HOME, "Downloads", "VTUGV_pointclouds_features")


def upsample_features(features, target_h, target_w):
    """
    Bilinear upsample patch-level features to pixel resolution.
    
    Args:
        features: (H_p, W_p, D) patch-level features
        target_h, target_w: target spatial dimensions
    
    Returns:
        (target_h, target_w, D) upsampled features
    """
    H_p, W_p, D = features.shape
    
    # Use numpy-based bilinear interpolation (avoid scipy/torch dependency)
    # Create coordinate grids
    y_out = np.linspace(0, H_p - 1, target_h)
    x_out = np.linspace(0, W_p - 1, target_w)
    
    # Floor and ceil indices
    y0 = np.floor(y_out).astype(int)
    y1 = np.minimum(y0 + 1, H_p - 1)
    x0 = np.floor(x_out).astype(int)
    x1 = np.minimum(x0 + 1, W_p - 1)
    
    # Fractional parts
    fy = y_out - y0
    fx = x_out - x0
    
    # Bilinear weights
    # Shape: (target_h, target_w, D)
    result = np.zeros((target_h, target_w, D), dtype=features.dtype)
    
    for yi in range(target_h):
        wy = fy[yi]
        for xi in range(target_w):
            wx = fx[xi]
            result[yi, xi] = (
                (1 - wy) * (1 - wx) * features[y0[yi], x0[xi]] +
                (1 - wy) * wx       * features[y0[yi], x1[xi]] +
                wy       * (1 - wx) * features[y1[yi], x0[xi]] +
                wy       * wx       * features[y1[yi], x1[xi]]
            )
    
    return result


def pca_project(features_flat, n_components=3):
    """
    Project high-dim features to n_components using PCA.
    Returns projected features normalized to [0, 1] per channel.
    """
    mean = features_flat.mean(axis=0)
    centered = features_flat - mean
    
    D = centered.shape[1]
    if D < centered.shape[0]:
        cov = centered.T @ centered / centered.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        top_k = eigenvectors[:, -n_components:][:, ::-1]
    else:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        top_k = Vt[:n_components].T
    
    projected = centered @ top_k  # (N, n_components)
    
    # Normalize each channel to [0, 1]
    for c in range(n_components):
        col = projected[:, c]
        mn, mx = col.min(), col.max()
        if mx > mn:
            projected[:, c] = (col - mn) / (mx - mn)
        else:
            projected[:, c] = 0.5
    
    return projected


def main():
    # ===== CONFIGURATION =====
    WINDOW_SIZE = 50
    SUBSAMPLE_RATE = 0.03   # 3% of pixels
    DEPTH_MIN_M = 0.5
    DEPTH_MAX_M = 50.0
    MAX_RENDER_DEPTH = 40.0
    # =========================

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cam_intrinsics = load_intrinsics()
    cam_extrinsics = load_extrinsics()

    # Load IMU and dead reckon
    imu_data = load_imu_data()
    dr_timestamps, dr_positions, dr_orientations = dead_reckon_positions(imu_data, subsample=10)

    # Gather images
    cam_files = {}
    for cam in CAMERAS:
        img_dir = os.path.join(IMAGE_DIR_BASE, cam)
        if not os.path.exists(img_dir):
            continue
        files = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg")) +
            glob.glob(os.path.join(img_dir, "*.png"))
        )
        cam_files[cam] = {os.path.splitext(os.path.basename(f))[0]: f for f in files}
        print(f"Found {len(cam_files[cam])} images for {cam}")

    if not cam_files:
        print("No images found.")
        return

    # Check which frames have features
    available_feature_frames = set()
    ref_cam = "front_left"
    if ref_cam in cam_files:
        feat_dir = os.path.join(FEATURES_DIR, ref_cam)
        if os.path.exists(feat_dir):
            feat_files = glob.glob(os.path.join(feat_dir, "*.npz"))
            for ff in feat_files:
                bn = os.path.splitext(os.path.basename(ff))[0]
                available_feature_frames.add(bn)
    
    print(f"Found {len(available_feature_frames)} frames with features for {ref_cam}")
    
    if not available_feature_frames:
        print("No feature extraction results found. Run estimate_features.py first.")
        return

    # Match frame tuples
    matched_frames = find_closest_matches(cam_files, tolerance_ns=50000000)
    print(f"Found {len(matched_frames)} synchronized frame tuples total.")
    
    # Filter to frames with features
    feature_frames = []
    for match in matched_frames:
        ref_name = match.get(ref_cam, "")
        if ref_name in available_feature_frames:
            feature_frames.append(match)
    
    print(f"Filtered to {len(feature_frames)} frames with features.")
    
    if not feature_frames:
        print("No matched frames have features.")
        return

    # Process in windows
    num_windows = (len(feature_frames) + WINDOW_SIZE - 1) // WINDOW_SIZE
    print(f"Processing {len(feature_frames)} frames in {num_windows} windows...")
    
    for win_idx in range(num_windows):
        start = win_idx * WINDOW_SIZE
        end = min(start + WINDOW_SIZE, len(feature_frames))
        window_frames = feature_frames[start:end]
        
        win_ref_name = window_frames[0].get(ref_cam, f"win_{win_idx}")
        print(f"\n=== Feature Window {win_idx+1}/{num_windows} ({len(window_frames)} frames) ===")
        
        # Reference pose
        first_ts = int(window_frames[0].get(ref_cam, "0"))
        T_world_ref = get_world_pose_at_timestamp(first_ts, dr_timestamps, dr_positions, dr_orientations)
        T_ref_world = np.linalg.inv(T_world_ref)
        
        window_points = []
        window_features = []
        
        for i, match in enumerate(window_frames):
            frame_ts = int(match.get(ref_cam, "0"))
            T_world_frame = get_world_pose_at_timestamp(frame_ts, dr_timestamps, dr_positions, dr_orientations)
            T_relative = T_ref_world @ T_world_frame
            
            for cam in CAMERAS:
                if cam not in match:
                    continue
                basename = match[cam]
                if basename not in cam_files.get(cam, {}):
                    continue
                
                depth_path = os.path.join(DEPTH_DIR_BASE, cam, basename + ".npy")
                feat_path = os.path.join(FEATURES_DIR, cam, basename + ".npz")
                
                if not os.path.exists(depth_path) or not os.path.exists(feat_path):
                    continue
                
                try:
                    # Load depth
                    depth = np.load(depth_path)
                    if depth.ndim == 3:
                        depth = depth.squeeze()
                    
                    # Load features
                    feat_data = np.load(feat_path)
                    features = feat_data["features"].astype(np.float32)  # (H_p, W_p, D)
                    orig_size = feat_data["orig_size"]  # [orig_h, orig_w]
                    
                    H, W = depth.shape
                    
                    # Depth normalization
                    depth_inv = 255.0 - depth
                    depth_norm = depth_inv / 255.0
                    depth_metric = DEPTH_MIN_M + depth_norm * (DEPTH_MAX_M - DEPTH_MIN_M)
                    
                    # Build subsample mask FIRST (before upsampling features for efficiency)
                    keep_mask = np.random.rand(H, W) < SUBSAMPLE_RATE
                    depth_valid = (depth_metric > DEPTH_MIN_M) & (depth_metric < MAX_RENDER_DEPTH)
                    valid_mask = depth_valid & keep_mask
                    
                    y_idxs, x_idxs = np.where(valid_mask)
                    z_vals = depth_metric[valid_mask]
                    
                    if len(z_vals) == 0:
                        continue
                    
                    # Map pixel coords to patch-level feature coords and interpolate
                    H_p, W_p, D = features.shape
                    # Scale pixel indices to patch grid coordinates
                    feat_y = y_idxs * (H_p - 1) / max(H - 1, 1)
                    feat_x = x_idxs * (W_p - 1) / max(W - 1, 1)
                    
                    # Bilinear interpolation at sampled points only (much faster than full upsample)
                    fy0 = np.floor(feat_y).astype(int)
                    fy1 = np.minimum(fy0 + 1, H_p - 1)
                    fx0 = np.floor(feat_x).astype(int)
                    fx1 = np.minimum(fx0 + 1, W_p - 1)
                    
                    wy = feat_y - fy0
                    wx = feat_x - fx0
                    
                    # (N, D) interpolated features
                    sampled_feats = (
                        (1 - wy)[:, None] * (1 - wx)[:, None] * features[fy0, fx0] +
                        (1 - wy)[:, None] * wx[:, None]       * features[fy0, fx1] +
                        wy[:, None]       * (1 - wx)[:, None] * features[fy1, fx0] +
                        wy[:, None]       * wx[:, None]       * features[fy1, fx1]
                    )
                    
                    # Intrinsics
                    if cam in cam_intrinsics:
                        fx_cam, fy_cam, cx, cy = cam_intrinsics[cam]
                    else:
                        fx_cam, fy_cam, cx, cy = DEFAULT_INTRINSICS
                    
                    # Backproject
                    x = (x_idxs - cx) * z_vals / fx_cam
                    y = (y_idxs - cy) * z_vals / fy_cam
                    points_cam = np.stack([x, y, z_vals], axis=1)
                    
                    # Camera -> Vehicle
                    if cam in cam_extrinsics:
                        T_veh_cam = cam_extrinsics[cam]
                    else:
                        T_veh_cam = np.eye(4)
                    
                    points_cam_h = np.column_stack((points_cam, np.ones(len(points_cam))))
                    points_veh = (T_veh_cam @ points_cam_h.T).T[:, :3]
                    
                    # Vehicle -> World (relative to window origin)
                    points_veh_h = np.column_stack((points_veh, np.ones(len(points_veh))))
                    points_world = (T_relative @ points_veh_h.T).T[:, :3]
                    
                    window_points.append(points_world)
                    window_features.append(sampled_feats)
                    
                except Exception as e:
                    print(f"  Error {cam}/{basename}: {e}")
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(window_frames)} frames...")
        
        if window_points:
            all_points = np.vstack(window_points)
            all_features = np.vstack(window_features)
            
            print(f"  Total: {len(all_points)} points × {all_features.shape[1]}-dim features")
            
            # PCA false-color for PLY visualization
            print(f"  Computing PCA projection for visualization...")
            pca_colors = pca_project(all_features, n_components=3)  # (N, 3) in [0,1]
            
            # Save PLY with PCA colors
            ply_path = os.path.join(OUTPUT_DIR, f"features_{win_ref_name}_w{WINDOW_SIZE}.ply")
            write_ply(ply_path, all_points, pca_colors)
            print(f"  Saved PLY: {ply_path}")
            
            # Save full features + points as NPZ for downstream
            npz_path = os.path.join(OUTPUT_DIR, f"features_{win_ref_name}_w{WINDOW_SIZE}.npz")
            np.savez_compressed(
                npz_path,
                points=all_points.astype(np.float32),
                features=all_features.astype(np.float16),  # Half precision to save space
            )
            npz_size = os.path.getsize(npz_path) / (1024 * 1024)
            print(f"  Saved NPZ: {npz_path} ({npz_size:.1f} MB)")
        else:
            print(f"  Window {win_idx+1} produced no points.")
    
    print("\nFeature fusion complete.")


if __name__ == "__main__":
    main()
