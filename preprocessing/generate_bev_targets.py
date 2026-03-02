import os
import glob
import numpy as np
import torch
from PIL import Image

# --- Configuration ---
HOME = os.path.expanduser("~")
DATASET_DIR = os.path.join(HOME, "Downloads", "Data_Outdoors")

# Directories for the Front Camera
SEMANTICS_DIR = os.path.join(DATASET_DIR, "semantics", "front_left")
# Directory where your raw LiDAR sweeps are stored (now in .npy format)
LIDAR_DIR = os.path.join(DATASET_DIR, "pointclouds") 
OUTPUT_DIR = os.path.join(DATASET_DIR, "bev_targets", "front_left")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# BEV Grid Configuration (Matches train_nano.py which expects 50x50)
BEV_RESOLUTION = 50  # 50x50 grid
# Define the physical area the UGV cares about (in meters)
# Example: 0 to +25m in front of the vehicle, and -12.5m to +12.5m left/right
Z_MIN, Z_MAX = 0.0, 25.0    # Forward distance
X_MIN, X_MAX = -12.5, 12.5  # Left/Right distance

# Note: Depending on your camera calibration, Y is usually 'down/up'. 
# We filter out points that are too high (e.g., branches) or too low.
Y_MIN, Y_MAX = -1.0, 2.0  # Height bounds relative to camera (in meters)

FX = 1037.350
FY = 1124.614
CX = 708.762
CY = 549.905
IMG_HEIGHT = 1920
IMG_WIDTH = 1200

# Extrinsics (Velodyne LiDAR to Camera)
# Derived from transformslid2left_cam.yaml (q=[0.5, -0.5, 0.5, -0.5], t=[0.355, -0.2, -0.275])
# The YAML forms T_camera_to_lidar. We need T_lidar_to_camera = inv(T_camera_to_lidar)
T_VELO_CAM = np.array([
    [ 0.0, -1.0,  0.0,  0.2   ],
    [ 0.0,  0.0, -1.0,  0.275 ],
    [ 1.0,  0.0,  0.0, -0.355 ],
    [ 0.0,  0.0,  0.0,  1.0   ]
], dtype=np.float32)

def generate_bev_for_frame():
    # 1. Get all semantic maps (images) and their timestamps
    sem_files = sorted(glob.glob(os.path.join(SEMANTICS_DIR, "*.npy")))
    
    if len(sem_files) == 0:
        print(f"No semantic maps found in {SEMANTICS_DIR}")
        return

    # 2. Get all LiDAR sweeps (now in .npy) and their timestamps
    lidar_files = sorted(glob.glob(os.path.join(LIDAR_DIR, "*.npy")))
    
    if len(lidar_files) == 0:
        print(f"No LiDAR Pointclouds found in {LIDAR_DIR}")
        return
        
    print(f"Found {len(sem_files)} Image frames and {len(lidar_files)} LiDAR sweeps. Generating BEV targets...")

    # Extract timestamps from filenames
    # Assuming the basename is the unix timestamp in nanoseconds, e.g. 1713882442722758732.npy
    lidar_times = []
    lidar_paths = []
    for lf in lidar_files:
        bn = os.path.splitext(os.path.basename(lf))[0]
        try:
            lidar_times.append(int(bn))
            lidar_paths.append(lf)
        except ValueError:
            pass
            
    lidar_times = np.array(lidar_times, dtype=np.int64)

    for sem_path in sem_files:
        basename = os.path.basename(sem_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        try:
            img_time = int(name_no_ext)
        except ValueError:
            print(f"Skipping {basename}, not a valid timestamp.")
            continue
        
        # 1. Load Semantic Map
        semantics = np.load(sem_path) # [H, W] class indices (0 to 12)
        H, W = semantics.shape
        
        # 2. Load Nearest LiDAR Point Cloud
        # Find the absolute time difference to all LiDAR sweeps
        time_diffs = np.abs(lidar_times - img_time)
        best_idx = np.argmin(time_diffs)
        
        # Optional: Skip if the nearest pointcloud is too far away in time (e.g. > 100ms sync error)
        # 1 second = 1,000,000,000 ns
        if time_diffs[best_idx] > 100_000_000: # 100ms
             print(f"LiDAR sync gap too large ({time_diffs[best_idx]/1e6:.1f}ms) for {name_no_ext}, skipping.")
             continue
             
        lidar_path = lidar_paths[best_idx]
            
        # Our .npy pointclouds likely are [N, 3] or [N, 4]
        points = np.load(lidar_path)
        if len(points.shape) > 1 and points.shape[1] >= 3:
            xyz_velo = points[:, :3].astype(np.float32) # [N, 3]
        else:
            print(f"Invalid pointcloud shape {points.shape} in {lidar_path}")
            continue
        
        # 3. Project LiDAR to Camera Frame
        # Convert to homogeneous coordinates
        N = xyz_velo.shape[0]
        xyz_velo_hom = np.hstack((xyz_velo, np.ones((N, 1)))) # [N, 4]
        
        # Transform to camera coordinate system
        xyz_cam_hom = (T_VELO_CAM @ xyz_velo_hom.T).T # [N, 4]
        xyz_cam = xyz_cam_hom[:, :3]
        
        X_c, Y_c, Z_c = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        
        # Filter points behind the camera
        front_cams = Z_c > 0.1
        X_c = X_c[front_cams]
        Y_c = Y_c[front_cams]
        Z_c = Z_c[front_cams]
        
        # 4. Project 3D Camera points to 2D Image Pixels
        u = (X_c * FX / Z_c) + CX
        v = (Y_c * FY / Z_c) + CY
        
        u = np.round(u).astype(np.int32)
        v = np.round(v).astype(np.int32)
        
        # Filter points outside image boundaries
        in_img = (u >= 0) & (u < IMG_WIDTH) & (v >= 0) & (v < IMG_HEIGHT)
        u_valid = u[in_img]
        v_valid = v[in_img]
        z_valid = Z_c[in_img]
        
        # 5. Create Dense Distance/Depth Image
        # Initialize with zeros
        lidar_depth_img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        lidar_valid_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=bool)
        
        # Populate depth image (handling multiple points hitting same pixel by keeping the closest)
        # Sort by depth descending so closest overwrites furthest
        sort_idx = np.argsort(z_valid)[::-1]
        u_valid = u_valid[sort_idx]
        v_valid = v_valid[sort_idx]
        z_valid = z_valid[sort_idx]
        
        lidar_depth_img[v_valid, u_valid] = z_valid
        lidar_valid_mask[v_valid, u_valid] = True
        
        # Resize to feature map size if you are doing depth distillation at feature scale
        # For our TrainEdge pipeline with DINOv2 small (patch size 14), feature h/w are H/14, W/14
        import cv2
        fH, fW = IMG_HEIGHT // 14, IMG_WIDTH // 14
        
        # Using nearest neighbor to avoid interpolating empty space with valid depth
        depth_2d_feat = cv2.resize(lidar_depth_img, (fW, fH), interpolation=cv2.INTER_NEAREST)
        valid_2d_feat = cv2.resize(lidar_valid_mask.astype(np.uint8), (fW, fH), interpolation=cv2.INTER_NEAREST).astype(bool)

        # 6. Save Explicit 2D Depth Targets for train_edge.py Method 2
        # We also need the full resolution depth map so occ_dataset.py can project Semantics & Features!
        out_path = os.path.join(OUTPUT_DIR, f"{name_no_ext}.npz")
        np.savez_compressed(
            out_path, 
            depth_map_full=lidar_depth_img,   # [256, 512] full res depth map
            gt_lidar_depth_2d=depth_2d_feat,  # [fH, fW] float32 depth in meters
            gt_lidar_valid=valid_2d_feat      # [fH, fW] boolean mask
        )

    print("Finished generating LiDAR 2D Depth targets!")

if __name__ == "__main__":
    generate_bev_for_frame()
