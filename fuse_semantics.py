"""
fuse_semantics.py
Lift 2D semantic segmentation labels into 3D point clouds using depth backprojection.
Colors points by semantic class (ADE20K palette).
Reuses IMU dead-reckoning and extrinsics from fuse_outdoors.py.
"""

import os
import glob
import numpy as np
from PIL import Image

# ---- Import shared functions from fuse_outdoors ----
from fuse_outdoors import (
    write_ply, load_intrinsics, load_extrinsics, load_imu_data,
    dead_reckon_positions, get_world_pose_at_timestamp,
    find_closest_matches, get_timestamp_from_filename,
    CAMERAS, DATASET_DIR, IMAGE_DIR_BASE, DEPTH_DIR_BASE,
    DEFAULT_INTRINSICS
)

# ---- Import custom classes and palette from estimate_semantics ----
from estimate_semantics import CUSTOM_CLASSES, CUSTOM_PALETTE

# Configuration
HOME = os.path.expanduser("~")
SEMANTICS_DIR = os.path.join(HOME, "Downloads", "the_great_outdoors_data_semantics")
OUTPUT_DIR = os.path.join(HOME, "Downloads", "VTUGV_pointclouds_semantics")

# Classes to EXCLUDE from 3D point cloud
# 7 = sky (doesn't make sense in 3D)
EXCLUDE_CLASSES = {7}


def main():
    # ===== CONFIGURATION =====
    WINDOW_SIZE = 50
    SUBSAMPLE_RATE = 0.05  # 5% — can be higher since semantic clouds are sparser
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

    # Check which frames have semantics available
    available_semantic_frames = set()
    ref_cam = "front_left"
    if ref_cam in cam_files:
        sem_dir = os.path.join(SEMANTICS_DIR, ref_cam)
        if os.path.exists(sem_dir):
            sem_files = glob.glob(os.path.join(sem_dir, "*.npy"))
            for sf in sem_files:
                bn = os.path.splitext(os.path.basename(sf))[0]
                available_semantic_frames.add(bn)
    
    print(f"Found {len(available_semantic_frames)} frames with semantic labels for {ref_cam}")
    
    if not available_semantic_frames:
        print("No semantic segmentation results found. Run estimate_semantics.py first.")
        return

    # Match frame tuples
    matched_frames = find_closest_matches(cam_files, tolerance_ns=50000000)
    print(f"Found {len(matched_frames)} synchronized frame tuples total.")
    
    # Filter to only frames that have semantic labels
    semantic_frames = []
    for match in matched_frames:
        ref_name = match.get(ref_cam, "")
        if ref_name in available_semantic_frames:
            semantic_frames.append(match)
    
    print(f"Filtered to {len(semantic_frames)} frames with semantic labels.")
    
    if not semantic_frames:
        print("No matched frames have semantic labels.")
        return

    # Process in windows
    num_windows = (len(semantic_frames) + WINDOW_SIZE - 1) // WINDOW_SIZE
    print(f"Processing {len(semantic_frames)} frames in {num_windows} windows...")
    
    for win_idx in range(num_windows):
        start = win_idx * WINDOW_SIZE
        end = min(start + WINDOW_SIZE, len(semantic_frames))
        window_frames = semantic_frames[start:end]
        
        win_ref_name = window_frames[0].get(ref_cam, f"win_{win_idx}")
        print(f"\n=== Semantic Window {win_idx+1}/{num_windows} ({len(window_frames)} frames) ===")
        
        # Reference pose for this window
        first_ts = int(window_frames[0].get(ref_cam, "0"))
        T_world_ref = get_world_pose_at_timestamp(first_ts, dr_timestamps, dr_positions, dr_orientations)
        T_ref_world = np.linalg.inv(T_world_ref)
        
        window_points = []
        window_colors = []  # Semantic colors from palette
        window_labels = []  # Raw class indices
        
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
                sem_path = os.path.join(SEMANTICS_DIR, cam, basename + ".npy")
                
                if not os.path.exists(depth_path) or not os.path.exists(sem_path):
                    continue
                
                try:
                    # Load depth
                    depth = np.load(depth_path)
                    if depth.ndim == 3:
                        depth = depth.squeeze()
                    
                    # Load semantic map
                    seg_map = np.load(sem_path)  # (H, W) uint8
                    
                    # Ensure same size
                    if depth.shape[:2] != seg_map.shape[:2]:
                        depth_pil = Image.fromarray(depth)
                        depth_pil = depth_pil.resize(
                            (seg_map.shape[1], seg_map.shape[0]),
                            Image.NEAREST
                        )
                        depth = np.array(depth_pil)
                    
                    H, W = depth.shape
                    
                    # Depth normalization
                    depth_inv = 255.0 - depth
                    depth_norm = depth_inv / 255.0
                    depth_metric = DEPTH_MIN_M + depth_norm * (DEPTH_MAX_M - DEPTH_MIN_M)
                    
                    # Build validity mask
                    keep_mask = np.random.rand(H, W) < SUBSAMPLE_RATE
                    depth_valid = (depth_metric > DEPTH_MIN_M) & (depth_metric < MAX_RENDER_DEPTH)
                    
                    # Exclude sky and other unwanted classes
                    class_valid = np.ones((H, W), dtype=bool)
                    for exc_cls in EXCLUDE_CLASSES:
                        class_valid &= (seg_map != exc_cls)
                    
                    valid_mask = depth_valid & class_valid & keep_mask
                    
                    y_idxs, x_idxs = np.where(valid_mask)
                    z_vals = depth_metric[valid_mask]
                    labels = seg_map[valid_mask]
                    
                    if len(z_vals) == 0:
                        continue
                    
                    # Map labels to colors
                    colors = CUSTOM_PALETTE[labels] / 255.0  # (N, 3) float
                    
                    # Intrinsics
                    if cam in cam_intrinsics:
                        fx, fy, cx, cy = cam_intrinsics[cam]
                    else:
                        fx, fy, cx, cy = DEFAULT_INTRINSICS
                    
                    # Backproject
                    x = (x_idxs - cx) * z_vals / fx
                    y = (y_idxs - cy) * z_vals / fy
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
                    window_colors.append(colors)
                    window_labels.append(labels)
                    
                except Exception as e:
                    print(f"  Error {cam}/{basename}: {e}")
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(window_frames)} frames...")
        
        if window_points:
            all_points = np.vstack(window_points)
            all_colors = np.vstack(window_colors)
            all_labels = np.concatenate(window_labels)
            
            # Save semantic point cloud (colored by class)
            ply_path = os.path.join(OUTPUT_DIR, f"semantic_{win_ref_name}_w{WINDOW_SIZE}.ply")
            write_ply(ply_path, all_points, all_colors)
            
            # Also save raw labels alongside for downstream use
            labels_path = os.path.join(OUTPUT_DIR, f"semantic_{win_ref_name}_w{WINDOW_SIZE}_labels.npy")
            np.save(labels_path, all_labels)
            
            # Print class distribution
            unique, counts = np.unique(all_labels, return_counts=True)
            print(f"  Saved {ply_path} ({len(all_points)} points)")
            print(f"  Class distribution:")
            for cls_id, count in zip(unique, counts):
                name = CUSTOM_CLASSES[cls_id] if cls_id < len(CUSTOM_CLASSES) else f"class_{cls_id}"
                pct = 100 * count / len(all_labels)
                print(f"    {name:20s}: {count:8d} ({pct:.1f}%)")
        else:
            print(f"  Window {win_idx+1} produced no points.")
    
    print("\nSemantic fusion complete.")


if __name__ == "__main__":
    main()
