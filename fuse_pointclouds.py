
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R
import bisect

# Configuration
DEPTH_DIR = r"data/depth"
IMAGE_DIR = r"data/images/multisense_left_image_rect_color"
SEMANTIC_DIR = r"data/segmentation/labels"
FEATURE_VIS_DIR = r"data/features/vis"
OUTPUT_DIR = r"data/pointclouds"
TIMESTAMPS_FILE = r"data/images/multisense_left_image_rect_color/timestamps.csv"
ODOM_FILE = r"data/odometry/odometry_filtered_odom.csv"

# Intrinsics (from data/calibration/multisense_intrinsics.txt)
FX = 477.6049499511719
FY = 477.6049499511719
CX = 499.5
CY = 252.0

# Extrinsics (Manually parsed from extrinsics.yaml for simplicity)
# vehicle -> multisense/head
T_VEHICLE_HEAD_TRANS = [-0.0949,  0.0641, -0.136]
T_VEHICLE_HEAD_QUAT = [-0.0119,  0.1131,  0.0003,  0.9935] # xyzw

# multisense/head -> multisense/left_camera_optical_frame
T_HEAD_CAM_TRANS = [0.005, 0.105, 0.000]
T_HEAD_CAM_QUAT = [-0.500, 0.500, -0.500, 0.500] # xyzw

# ADE20K Colormap
np.random.seed(42)
SEMANTIC_COLORMAP = np.random.randint(0, 255, (256, 3), dtype=np.uint8)

def get_transform_matrix(translation, quaternion):
    """
    Convert translation and quaternion (xyzw) to 4x4 matrix.
    """
    mat = np.eye(4)
    mat[:3, 3] = translation
    rot = R.from_quat(quaternion).as_matrix()
    mat[:3, :3] = rot
    return mat

def load_timestamps(csv_path):
    df = pd.read_csv(csv_path)
    # Combine sec and nsec
    timestamps = df['timestamp_sec'] + df['timestamp_nsec'] * 1e-9
    frame_indices = df['frame_index'].values
    filenames = df['filename'].values
    return timestamps.values, frame_indices, filenames

def load_odometry(csv_path):
    df = pd.read_csv(csv_path)
    timestamps = df['timestamp_sec'] + df['timestamp_nsec'] * 1e-9
    poses = []
    for _, row in df.iterrows():
        trans = [row['pos_x'], row['pos_y'], row['pos_z']]
        quat = [row['orient_x'], row['orient_y'], row['orient_z'], row['orient_w']]
        poses.append(get_transform_matrix(trans, quat))
    return timestamps.values, np.array(poses)

def find_nearest_pose(target_time, odom_times, odom_poses):
    idx = bisect.bisect_left(odom_times, target_time)
    if idx == 0:
        return odom_poses[0]
    if idx == len(odom_times):
        return odom_poses[-1]
    
    # Linear check for closest
    dt_before = abs(target_time - odom_times[idx-1])
    dt_after = abs(target_time - odom_times[idx])
    
    if dt_before < dt_after:
        return odom_poses[idx-1]
    else:
        return odom_poses[idx]

def save_ply(output_path, points, colors):
    print(f"Saving {len(points)} points to {output_path}...")
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    data = np.hstack([points, colors.astype(np.float32)])
    np.savetxt(output_path, data, fmt="%.4f %.4f %.4f %d %d %d", header=header, comments='')

def main():
    # Setup Static Transforms
    # NOTE: "from_frame: vehicle, to_frame: multisense/head" usually means transform of Head IN Vehicle frame (T_veh_head)
    # If we want to transform Point_Head to Point_Vehicle, we multiply T_veh_head * P_head
    
    # T_vehicle_head
    T_vh = get_transform_matrix(T_VEHICLE_HEAD_TRANS, T_VEHICLE_HEAD_QUAT)
    
    # T_head_cam
    T_hc = get_transform_matrix(T_HEAD_CAM_TRANS, T_HEAD_CAM_QUAT)
    
    # Full static transform: P_vehicle = T_vh * T_hc * P_cam
    T_static = T_vh @ T_hc

    # Load Timestamps and Odom
    print("Loading metadata...")
    img_times, _, img_filenames = load_timestamps(TIMESTAMPS_FILE)
    odom_times, odom_poses = load_odometry(ODOM_FILE)
    
    # Buffers for fusion
    fused_points = []
    fused_colors_rgb = []
    fused_colors_sem = []
    fused_colors_feat = []
    
    # Process all frames
    NUM_FRAMES_TO_FUSE = len(img_filenames)
    
    print(f"Fusing all {NUM_FRAMES_TO_FUSE} frames...")
    
    for i in range(NUM_FRAMES_TO_FUSE):
        filename = img_filenames[i]
        basename = os.path.splitext(filename)[0]
        frame_id_str = basename.replace("frame_", "")
        
        print(f"[{i+1}/{NUM_FRAMES_TO_FUSE}] Processing {filename}...")
        
        # Load Data
        depth_path = os.path.join(DEPTH_DIR, f"depth_{frame_id_str}.npy")
        img_path = os.path.join(IMAGE_DIR, filename)
        sem_path = os.path.join(SEMANTIC_DIR, filename)
        feat_path = os.path.join(FEATURE_VIS_DIR, f"{basename}_pca.png")
        
        if not os.path.exists(depth_path): continue
        if not os.path.exists(img_path): continue
        
        depth = np.load(depth_path).astype(np.float32)
        if depth.ndim == 3: depth = depth.squeeze()
        
        rgb_img = Image.open(img_path).convert("RGB")
        rgb_arr = np.array(rgb_img)
        
        sem_img = Image.open(sem_path) if os.path.exists(sem_path) else None
        feat_img = Image.open(feat_path).convert("RGB") if os.path.exists(feat_path) else None
        if feat_img and feat_img.size != rgb_img.size: feat_img = feat_img.resize(rgb_img.size, Image.NEAREST)
        
        # Backproject
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Filter
        mask = (depth > 1.0) & (depth < 50.0) # Filter near/far clipping
        mask = mask & (np.random.rand(H, W) < 0.1) # DOWNSAMPLE 10% for fusion (otherwise too huge)
        
        u = u[mask]
        v = v[mask]
        z = depth[mask]
        
        x = (u - CX) * z / FX
        y = (v - CY) * z / FY
        
        # Points in Camera Frame (N, 3)
        points_cam = np.stack([x, y, z], axis=1)
        ones = np.ones((points_cam.shape[0], 1))
        points_cam_hom = np.hstack([points_cam, ones]) # (N, 4)
        
        # Get Dynamic Transform (Vehicle -> World)
        t_img = img_times[i]
        T_vw = find_nearest_pose(t_img, odom_times, odom_poses)
        
        # Full Transform: P_world = T_vw * T_vh * T_hc * P_cam
        #               = T_vw * T_static * P_cam
        
        T_full = T_vw @ T_static
        
        # Transform Points
        points_world_hom = (T_full @ points_cam_hom.T).T # (4, 4) @ (4, N) -> (4, N) -> (N, 4)
        points_world = points_world_hom[:, :3]
        
        fused_points.append(points_world)
        
        # Color: RGB
        fused_colors_rgb.append(rgb_arr[mask])
        
        # Color: Semantic
        if sem_img:
            sem_arr = np.array(sem_img)
            sem_color = SEMANTIC_COLORMAP[sem_arr]
            fused_colors_sem.append(sem_color[mask])
        else:
            fused_colors_sem.append(np.zeros_like(rgb_arr[mask]))

        # Color: Features
        if feat_img:
            feat_arr = np.array(feat_img)
            fused_colors_feat.append(feat_arr[mask])
        else:
            fused_colors_feat.append(np.zeros_like(rgb_arr[mask]))

    # Concatenate
    if not fused_points:
        print("No points fused.")
        return

    all_points = np.vstack(fused_points)
    all_rgb = np.vstack(fused_colors_rgb)
    all_sem = np.vstack(fused_colors_sem)
    all_feat = np.vstack(fused_colors_feat)
    
    print(f"Total fused points: {len(all_points)}")
    
    # Save
    save_ply(os.path.join(OUTPUT_DIR, "fused_map_rgb.ply"), all_points, all_rgb)
    save_ply(os.path.join(OUTPUT_DIR, "fused_map_semantic.ply"), all_points, all_sem)
    save_ply(os.path.join(OUTPUT_DIR, "fused_map_features.ply"), all_points, all_feat)
    print("Fusion complete.")

if __name__ == "__main__":
    main()
