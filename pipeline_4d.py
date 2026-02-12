"""
=============================================================================
Self-Supervised 4D Occupancy Projection Pipeline
=============================================================================
TartanDrive 2.0 — QueryOcc-style photometric consistency training.

Pipeline:  RGB image → stereo depth → backproject to 3D → transform via
           TartanVO pose → reproject into neighbor camera → photometric loss.

Each section below maps to a cell in new_notebook.ipynb (Sections 0-5).
=============================================================================
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 0 — Imports and Dependencies                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import csv
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

print("✓ All imports successful")
print(f"  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — Configuration & Path Setup                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED_DIR = BASE_DIR / "extracted"

IMAGE_DIR = EXTRACTED_DIR / "images" / "multisense_left_image_rect_color"
DEPTH_DIR = EXTRACTED_DIR / "depth"
ODOM_DIR  = EXTRACTED_DIR / "odometry"
CALIB_DIR = EXTRACTED_DIR / "calibration"
DEBUG_DIR = EXTRACTED_DIR / "projection_debug"

DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ── Training hyperparameters ─────────────────────────────────────────────
LEARNING_RATE  = 5e-4
NUM_EPOCHS     = 30
BATCH_SIZE     = 2
IMG_HEIGHT     = 544
IMG_WIDTH      = 1024

# ── Voxel grid parameters (for future QueryOcc) ─────────────────────────
VOXEL_X_RANGE  = (-10.0, 10.0)   # meters, lateral
VOXEL_Y_RANGE  = (-3.0,  3.0)    # meters, vertical
VOXEL_Z_RANGE  = (0.5,   50.0)   # meters, forward (depth)
VOXEL_SIZE     = 0.5             # meters per voxel

print(f"✓ Configuration loaded")
print(f"  Base dir:   {BASE_DIR}")
print(f"  Image dir:  {IMAGE_DIR}")
print(f"  Depth dir:  {DEPTH_DIR}")
print(f"  Odom dir:   {ODOM_DIR}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — Data Calibration                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_intrinsics(filepath):
    """
    Parse multisense_intrinsics.txt to extract the camera intrinsic matrix K
    and projection matrix P for the left color camera.

    Returns:
        K:  (3, 3) intrinsic matrix
        P:  (3, 4) projection matrix
        D:  distortion coefficients
        img_size: (width, height)
    """
    with open(filepath, "r") as f:
        content = f.read()

    # We want the left_image_rect_color section
    sections = content.split("---")
    target_section = None
    for section in sections:
        if "left_image_rect_color" in section or "left/camera_info" in section:
            # Prefer left_image_rect_color if present
            if "left_image_rect_color" in section:
                target_section = section
                break
            elif target_section is None:
                target_section = section

    if target_section is None:
        target_section = sections[0]  # fallback to first section

    # Parse K matrix (3x3 stored as flat list of 9)
    for line in target_section.split("\n"):
        line = line.strip()
        if line.startswith("K:"):
            k_str = line[2:].strip()
            k_vals = [float(x) for x in k_str.strip("[]").split(",")]
            K = np.array(k_vals).reshape(3, 3)
        elif line.startswith("P:"):
            p_str = line[2:].strip()
            p_vals = [float(x) for x in p_str.strip("[]").split(",")]
            P = np.array(p_vals).reshape(3, 4)
        elif line.startswith("D:"):
            d_str = line[2:].strip()
            d_vals = [float(x) for x in d_str.strip("[]").split(",")]
            D = np.array(d_vals)
        elif line.startswith("width:"):
            width = int(line.split(":")[1].strip())
        elif line.startswith("height:"):
            height = int(line.split(":")[1].strip())

    return K, P, D, (width, height)


def parse_extrinsics(filepath):
    """
    Parse extrinsics.yaml to build transformation matrices.

    Returns dict of transform name -> 4x4 homogeneous matrix.
    Key transforms:
      - 'vehicle_to_multisense_head':  T from vehicle frame to camera head
      - 'multisense_head_to_left_optical': T from head to left camera optical frame
      - 'vehicle_to_left_optical': composed full chain
    """
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    transforms = {}
    for entry in data["transform_params"]:
        t = np.array(entry["translation"])
        q = entry["quaternion"]  # [x, y, z, w]
        R = Rotation.from_quat(q).as_matrix()  # scipy uses [x,y,z,w]

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        key = f"{entry['from_frame']}_to_{entry['to_frame']}"
        # Clean up slashes in frame names
        key = key.replace("/", "_").replace("multisense_", "ms_")
        transforms[key] = T

    # Compose vehicle → left_camera_optical_frame
    # Chain: vehicle → multisense/head → multisense/left_camera_optical_frame
    T_v2head = None
    T_head2optical = None

    for entry in data["transform_params"]:
        if entry["from_frame"] == "vehicle" and "head" in entry["to_frame"]:
            t = np.array(entry["translation"])
            q = entry["quaternion"]
            R = Rotation.from_quat(q).as_matrix()
            T_v2head = np.eye(4)
            T_v2head[:3, :3] = R
            T_v2head[:3, 3] = t

        if "head" in entry["from_frame"] and "left_camera_optical" in entry["to_frame"]:
            t = np.array(entry["translation"])
            q = entry["quaternion"]
            R = Rotation.from_quat(q).as_matrix()
            T_head2optical = np.eye(4)
            T_head2optical[:3, :3] = R
            T_head2optical[:3, 3] = t

    if T_v2head is not None and T_head2optical is not None:
        T_vehicle_to_cam = T_head2optical @ T_v2head
        transforms["vehicle_to_left_optical"] = T_vehicle_to_cam
    else:
        print("  WARNING: Could not compose full vehicle→camera chain")
        # Identity fallback
        transforms["vehicle_to_left_optical"] = np.eye(4)

    return transforms


def parse_odometry(filepath):
    """
    Parse TartanVO odometry CSV.

    Returns:
        timestamps: list of float (seconds.nanoseconds)
        poses: list of 4x4 numpy arrays (vehicle-to-world)
    """
    timestamps = []
    poses = []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["timestamp_sec"]) + float(row["timestamp_nsec"]) * 1e-9
            timestamps.append(ts)

            pos = np.array([
                float(row["pos_x"]),
                float(row["pos_y"]),
                float(row["pos_z"])
            ])
            quat = np.array([
                float(row["orient_x"]),
                float(row["orient_y"]),
                float(row["orient_z"]),
                float(row["orient_w"])
            ])

            R = Rotation.from_quat(quat).as_matrix()  # [x,y,z,w] format
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = pos

            poses.append(T)

    return np.array(timestamps), poses


def parse_image_timestamps(filepath):
    """
    Parse the image timestamps CSV.

    Returns:
        frame_indices: list of int
        timestamps: list of float (seconds.nanoseconds)
        filenames: list of str
    """
    frame_indices = []
    timestamps = []
    filenames = []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_indices.append(int(row["frame_index"]))
            ts = float(row["timestamp_sec"]) + float(row["timestamp_nsec"]) * 1e-9
            timestamps.append(ts)
            filenames.append(row["filename"])

    return frame_indices, np.array(timestamps), filenames


def match_poses_to_frames(image_timestamps, odom_timestamps, odom_poses):
    """
    For each image timestamp, find the nearest odometry pose.

    Returns:
        matched_poses: list of 4x4 arrays (one per image frame)
        time_offsets: list of float (time difference in seconds)
    """
    matched_poses = []
    time_offsets = []

    for img_ts in image_timestamps:
        diffs = np.abs(odom_timestamps - img_ts)
        best_idx = np.argmin(diffs)
        matched_poses.append(odom_poses[best_idx])
        time_offsets.append(float(diffs[best_idx]))

    return matched_poses, time_offsets


# ── Run calibration ──────────────────────────────────────────────────────
print("\n── Section 2: Data Calibration ──")

K, P, D, img_size = parse_intrinsics(CALIB_DIR / "multisense_intrinsics.txt")
print(f"  Intrinsics K:\n{K}")
print(f"  Image size: {img_size}")

extrinsics = parse_extrinsics(CALIB_DIR / "extrinsics.yaml")
T_vehicle_to_cam = extrinsics["vehicle_to_left_optical"]
T_cam_to_vehicle = np.linalg.inv(T_vehicle_to_cam)
print(f"  T_vehicle_to_cam (shape {T_vehicle_to_cam.shape}):")
print(f"  {T_vehicle_to_cam}")

odom_timestamps, odom_poses = parse_odometry(ODOM_DIR / "tartanvo_odom.csv")
print(f"  Loaded {len(odom_poses)} odometry poses")
print(f"  Odom time range: {odom_timestamps[0]:.3f} → {odom_timestamps[-1]:.3f}s")

frame_indices, img_timestamps, img_filenames = parse_image_timestamps(
    IMAGE_DIR / "timestamps.csv"
)
print(f"  Loaded {len(frame_indices)} image timestamps")

matched_poses, time_offsets = match_poses_to_frames(
    img_timestamps, odom_timestamps, odom_poses
)
max_offset = max(time_offsets)
mean_offset = np.mean(time_offsets)
print(f"  Pose matching: mean offset={mean_offset*1000:.1f}ms, max={max_offset*1000:.1f}ms")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — Dataset & DataLoader                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TartanDrive4DDataset(Dataset):
    """
    Dataset returning temporal triplets (T-1, T, T+1) with depth and poses.

    Each sample provides everything needed for the photometric consistency
    forward pass:
      - Three consecutive RGB frames (prev, curr, next)
      - Depth map for the current frame
      - 4x4 world poses for all three frames
      - Camera intrinsic matrix K
    """

    def __init__(self, image_dir, depth_dir, frame_indices, filenames,
                 matched_poses, img_timestamps, K,
                 img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
        self.image_dir = Path(image_dir)
        self.depth_dir = Path(depth_dir)
        self.img_height = img_height
        self.img_width = img_width

        # K as torch tensor (3x3)
        self.K = torch.from_numpy(K.astype(np.float32))
        self.K_inv = torch.inverse(self.K)

        # Build valid triplets: need frames at index i-1, i, i+1
        # Also need depth for frame i  and valid poses for all three
        self.triplets = []
        for i in range(1, len(frame_indices) - 1):
            # Check depth file exists for current frame
            depth_path = self.depth_dir / f"depth_{frame_indices[i]:06d}.npy"
            if not depth_path.exists():
                continue

            # Check that timestamps are reasonably close (reject large gaps)
            dt_prev = img_timestamps[i] - img_timestamps[i - 1]
            dt_next = img_timestamps[i + 1] - img_timestamps[i]
            if dt_prev > 0.5 or dt_next > 0.5:  # skip if gap > 500ms
                continue

            self.triplets.append({
                "prev_img": str(self.image_dir / filenames[i - 1]),
                "curr_img": str(self.image_dir / filenames[i]),
                "next_img": str(self.image_dir / filenames[i + 1]),
                "depth":    str(depth_path),
                "pose_prev": matched_poses[i - 1],
                "pose_curr": matched_poses[i],
                "pose_next": matched_poses[i + 1],
                "timestamp": img_timestamps[i],
                "frame_idx": frame_indices[i],
            })

        print(f"  TartanDrive4DDataset: {len(self.triplets)} valid triplets "
              f"from {len(frame_indices)} frames")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        sample = self.triplets[idx]

        # Load images (BGR → RGB, normalize to [0, 1])
        img_prev = cv2.imread(sample["prev_img"])
        img_curr = cv2.imread(sample["curr_img"])
        img_next = cv2.imread(sample["next_img"])

        img_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # (H, W, 3) → (3, H, W)
        img_prev = torch.from_numpy(img_prev).permute(2, 0, 1)
        img_curr = torch.from_numpy(img_curr).permute(2, 0, 1)
        img_next = torch.from_numpy(img_next).permute(2, 0, 1)

        # Load depth (float32, meters; 0 = invalid)
        depth = np.load(sample["depth"]).astype(np.float32)
        depth = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)

        # Poses as 4x4 tensors
        pose_prev = torch.from_numpy(sample["pose_prev"].astype(np.float32))
        pose_curr = torch.from_numpy(sample["pose_curr"].astype(np.float32))
        pose_next = torch.from_numpy(sample["pose_next"].astype(np.float32))

        return {
            "img_prev": img_prev,
            "img_curr": img_curr,
            "img_next": img_next,
            "depth":    depth,
            "pose_prev": pose_prev,
            "pose_curr": pose_curr,
            "pose_next": pose_next,
            "K":        self.K,
            "K_inv":    self.K_inv,
            "timestamp": sample["timestamp"],
            "frame_idx": sample["frame_idx"],
        }


# ── Create dataset and dataloader ────────────────────────────────────────
print("\n── Section 3: Dataset & DataLoader ──")

dataset = TartanDrive4DDataset(
    image_dir=IMAGE_DIR,
    depth_dir=DEPTH_DIR,
    frame_indices=frame_indices,
    filenames=img_filenames,
    matched_poses=matched_poses,
    img_timestamps=img_timestamps,
    K=K,
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Quick sanity check
sample = dataset[0]
print(f"  Sample shapes:")
print(f"    img_curr:  {sample['img_curr'].shape}")
print(f"    depth:     {sample['depth'].shape}")
print(f"    pose_curr: {sample['pose_curr'].shape}")
print(f"    K:         {sample['K'].shape}")
print(f"    timestamp: {sample['timestamp']:.6f}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — Projection (the 4D pipeline core)                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
#  The 4D projection pipeline:
#
#    P_world = T_v2w · T_c2v · (D · K⁻¹ · [u, v, 1]ᵀ)
#
#  For photometric consistency we don't need to go all the way to world
#  coordinates. Instead we compute the RELATIVE transform between two
#  camera frames and warp one image into the other's viewpoint:
#
#    1. Backproject current depth map to 3D points in camera frame
#    2. Transform 3D points from current camera to neighbor camera
#    3. Project transformed points into the neighbor's image plane
#    4. Sample the neighbor image at the projected coordinates
#    5. Compare sampled image with actual current image → loss
#


def backproject_depth_to_3d(depth, K_inv):
    """
    Backproject a depth map into 3D points in the camera coordinate frame.

    Args:
        depth: (B, 1, H, W) depth map in meters
        K_inv: (B, 3, 3)    inverse intrinsic matrix

    Returns:
        points_3d: (B, 3, H*W) 3D points in camera frame
    """
    B, _, H, W = depth.shape

    # Create pixel coordinate grid
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=depth.device),
        torch.arange(W, dtype=torch.float32, device=depth.device),
        indexing="ij"
    )
    ones = torch.ones_like(u)

    # Homogeneous pixel coordinates: (3, H*W)
    pixel_coords = torch.stack([u, v, ones], dim=0).reshape(3, -1)  # (3, H*W)
    pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1)       # (B, 3, H*W)

    # Normalized camera rays: K⁻¹ · [u, v, 1]ᵀ
    cam_rays = K_inv @ pixel_coords  # (B, 3, H*W)

    # Scale by depth: D · K⁻¹ · [u, v, 1]ᵀ
    depth_flat = depth.reshape(B, 1, H * W)  # (B, 1, H*W)
    points_3d = cam_rays * depth_flat         # (B, 3, H*W)

    return points_3d


def transform_points_3d(points_3d, T):
    """
    Apply a 4×4 rigid transform to 3D points.

    Args:
        points_3d: (B, 3, N)  3D points
        T:         (B, 4, 4)  transformation matrix

    Returns:
        transformed: (B, 3, N) transformed 3D points
    """
    B, _, N = points_3d.shape
    R = T[:, :3, :3]  # (B, 3, 3)
    t = T[:, :3, 3:]  # (B, 3, 1)

    transformed = R @ points_3d + t  # (B, 3, N)
    return transformed


def project_3d_to_2d(points_3d, K):
    """
    Project 3D points onto the image plane.

    Args:
        points_3d: (B, 3, N) 3D points in camera frame
        K:         (B, 3, 3) intrinsic matrix

    Returns:
        pixel_coords: (B, 2, N) pixel coordinates (u, v)
        valid_mask:    (B, N)    True where depth > 0 (in front of camera)
    """
    # Project: K · P_3d → [u*z, v*z, z]
    projected = K @ points_3d  # (B, 3, N)

    # Avoid division by zero
    z = projected[:, 2:3, :]  # (B, 1, N)
    z = z.clamp(min=1e-6)

    pixel_coords = projected[:, :2, :] / z  # (B, 2, N)

    # Valid if z > 0 (point is in front of the camera)
    valid_mask = (points_3d[:, 2, :] > 0.1)  # (B, N)

    return pixel_coords, valid_mask


def compute_relative_transform(T_src, T_tgt, T_cam_to_vehicle_torch, T_vehicle_to_cam_torch):
    """
    Compute the relative camera-to-camera transform between two frames.

    The full chain:
      P_cam_tgt = T_v2c · T_world2v_tgt · T_v2world_src · T_c2v · P_cam_src

    Simplified: T_rel = T_v2c · inv(T_tgt) · T_src · T_c2v

    Args:
        T_src: (B, 4, 4) source frame world pose (vehicle-to-world)
        T_tgt: (B, 4, 4) target frame world pose (vehicle-to-world)
        T_cam_to_vehicle_torch: (4, 4) static camera-to-vehicle transform
        T_vehicle_to_cam_torch: (4, 4) static vehicle-to-camera transform

    Returns:
        T_rel: (B, 4, 4) relative transform: src_cam → tgt_cam
    """
    B = T_src.shape[0]

    T_c2v = T_cam_to_vehicle_torch.unsqueeze(0).expand(B, -1, -1)  # (B, 4, 4)
    T_v2c = T_vehicle_to_cam_torch.unsqueeze(0).expand(B, -1, -1)  # (B, 4, 4)

    # inv(T_tgt): world-to-vehicle of target
    T_tgt_inv = torch.inverse(T_tgt)

    # Relative transform: src_cam → src_vehicle → world → tgt_vehicle → tgt_cam
    T_rel = T_v2c @ T_tgt_inv @ T_src @ T_c2v

    return T_rel


def warp_image(src_img, tgt_depth, K, K_inv, T_tgt_to_src):
    """
    Warp a source image into the target camera's viewpoint using the
    target's depth map and the relative camera transform.

    This is the core of photometric consistency:
      1. Backproject target depth → 3D points in target camera frame
      2. Transform to source camera frame via T_tgt_to_src
      3. Project onto source image plane
      4. Sample source image at projected coordinates

    Args:
        src_img:       (B, 3, H, W)  source RGB image
        tgt_depth:     (B, 1, H, W)  target depth map
        K:             (B, 3, 3)     intrinsic matrix
        K_inv:         (B, 3, 3)     inverse intrinsic matrix
        T_tgt_to_src:  (B, 4, 4)     transform from target cam to source cam

    Returns:
        warped_img:    (B, 3, H, W)  source image warped to target viewpoint
        valid_mask:    (B, 1, H, W)  mask of valid projections
    """
    B, _, H, W = src_img.shape

    # Step 1: Backproject target depth to 3D
    pts_3d = backproject_depth_to_3d(tgt_depth, K_inv)  # (B, 3, H*W)

    # Step 2: Transform to source camera frame
    pts_in_src = transform_points_3d(pts_3d, T_tgt_to_src)  # (B, 3, H*W)

    # Step 3: Project onto source image plane
    pixel_coords, valid = project_3d_to_2d(pts_in_src, K)  # (B, 2, H*W), (B, H*W)

    # Step 4: Normalize to [-1, 1] for grid_sample
    u = pixel_coords[:, 0, :]  # (B, H*W)
    v = pixel_coords[:, 1, :]  # (B, H*W)

    u_norm = 2.0 * u / (W - 1) - 1.0
    v_norm = 2.0 * v / (H - 1) - 1.0

    # Also mark out-of-bounds pixels as invalid
    valid = valid & (u_norm > -1) & (u_norm < 1) & (v_norm > -1) & (v_norm < 1)

    # Reshape for grid_sample: (B, H, W, 2)
    grid = torch.stack([u_norm, v_norm], dim=-1).reshape(B, H, W, 2)

    # Step 5: Sample source image
    warped_img = F.grid_sample(
        src_img, grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )

    valid_mask = valid.reshape(B, 1, H, W).float()

    # Also mask out pixels with zero depth (no valid depth)
    depth_valid = (tgt_depth > 0).float()
    valid_mask = valid_mask * depth_valid

    return warped_img, valid_mask


print("\n── Section 4: Projection Functions Defined ──")

# ── Convert static transforms to torch ───────────────────────────────────
T_cam_to_vehicle_torch = torch.from_numpy(T_cam_to_vehicle.astype(np.float32)).to(DEVICE)
T_vehicle_to_cam_torch = torch.from_numpy(T_vehicle_to_cam.astype(np.float32)).to(DEVICE)

# ── Sanity check: warp frame 5 using frame 4's image ─────────────────────
print("\n  Running projection sanity check...")

sample = dataset[5] if len(dataset) > 5 else dataset[0]
img_src = sample["img_prev"].unsqueeze(0).to(DEVICE)   # (1, 3, H, W) — neighbor
img_tgt = sample["img_curr"].unsqueeze(0).to(DEVICE)   # (1, 3, H, W) — current
depth   = sample["depth"].unsqueeze(0).to(DEVICE)       # (1, 1, H, W)
K_b     = sample["K"].unsqueeze(0).to(DEVICE)           # (1, 3, 3)
K_inv_b = sample["K_inv"].unsqueeze(0).to(DEVICE)       # (1, 3, 3)
pose_src = sample["pose_prev"].unsqueeze(0).to(DEVICE)  # (1, 4, 4)
pose_tgt = sample["pose_curr"].unsqueeze(0).to(DEVICE)  # (1, 4, 4)

# Relative transform: target_cam → source_cam
T_tgt_to_src = compute_relative_transform(
    pose_tgt, pose_src,
    T_cam_to_vehicle_torch, T_vehicle_to_cam_torch
)

warped, mask = warp_image(img_src, depth, K_b, K_inv_b, T_tgt_to_src)

# Save debug visualization
fig, axes = plt.subplots(2, 2, figsize=(20, 10))
axes[0, 0].imshow(img_tgt[0].cpu().permute(1, 2, 0).numpy())
axes[0, 0].set_title("Target frame (current)")
axes[0, 1].imshow(img_src[0].cpu().permute(1, 2, 0).numpy())
axes[0, 1].set_title("Source frame (previous)")
axes[1, 0].imshow(warped[0].cpu().detach().permute(1, 2, 0).numpy().clip(0, 1))
axes[1, 0].set_title("Warped source → target viewpoint")
axes[1, 1].imshow(mask[0, 0].cpu().detach().numpy(), cmap="gray")
axes[1, 1].set_title("Valid projection mask")
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
debug_path = DEBUG_DIR / "projection_sanity_check.png"
plt.savefig(debug_path, dpi=100, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved projection debug image: {debug_path}")
print(f"    Valid pixels: {mask.sum().item():.0f} / {mask.numel()}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — Loss Functions & Training Loop                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def ssim(x, y, window_size=3):
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    Args:
        x, y: (B, 3, H, W) images in [0, 1]
        window_size: size of the averaging window

    Returns:
        ssim_map: (B, 1, H, W) per-pixel SSIM (1 = identical)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    pad = window_size // 2

    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(x ** 2, window_size, stride=1, padding=pad) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y ** 2, window_size, stride=1, padding=pad) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_xy

    ssim_num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ssim_den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = ssim_num / ssim_den
    return ssim_map.mean(dim=1, keepdim=True)  # average over RGB → (B, 1, H, W)


def photometric_loss(pred_img, target_img, valid_mask, alpha=0.85):
    """
    Combined photometric reconstruction loss: α·SSIM + (1-α)·L1.

    Args:
        pred_img:    (B, 3, H, W) warped/reconstructed image
        target_img:  (B, 3, H, W) ground truth image
        valid_mask:  (B, 1, H, W) valid pixel mask
        alpha:       weight for SSIM vs L1

    Returns:
        loss: scalar photometric loss
    """
    # L1 loss
    l1 = (pred_img - target_img).abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)

    # SSIM loss (1 - SSIM, so lower is better)
    ssim_val = ssim(pred_img, target_img)
    ssim_loss = (1.0 - ssim_val) / 2.0  # scale to [0, 1]

    # Combined
    combined = alpha * ssim_loss + (1.0 - alpha) * l1

    # Apply mask: only count valid pixels
    masked = combined * valid_mask
    if valid_mask.sum() > 0:
        loss = masked.sum() / valid_mask.sum()
    else:
        loss = masked.sum()  # will be 0

    return loss


def smoothness_loss(depth, image):
    """
    Edge-aware depth smoothness loss.
    Depth gradients are penalized less at image edges (likely object boundaries).

    Args:
        depth: (B, 1, H, W) depth or inverse depth
        image: (B, 3, H, W) RGB image

    Returns:
        loss: scalar smoothness loss
    """
    # Depth gradients
    grad_depth_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    grad_depth_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

    # Image gradients (for edge-awareness)
    grad_img_x = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]).mean(dim=1, keepdim=True)
    grad_img_y = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(dim=1, keepdim=True)

    # Weight depth smoothness by exp(-image_gradient)
    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)

    return grad_depth_x.mean() + grad_depth_y.mean()


# ── Training Loop ────────────────────────────────────────────────────────
print("\n── Section 5: Loss Functions & Training ──")



class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNetEncoder(nn.Module):
    """Pretrained ResNet-18 feature extractor at 4 scales.

    Scale 0: 64ch,  H/2  × W/2
    Scale 1: 64ch,  H/4  × W/4
    Scale 2: 128ch, H/8  × W/8
    Scale 3: 256ch, H/16 × W/16
    """

    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool   = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.num_ch_enc = [64, 64, 128, 256]

    def forward(self, x):
        f0 = self.layer0(x)               # (B, 64,  H/2,  W/2)
        f1 = self.layer1(self.pool(f0))    # (B, 64,  H/4,  W/4)
        f2 = self.layer2(f1)               # (B, 128, H/8,  W/8)
        f3 = self.layer3(f2)               # (B, 256, H/16, W/16)
        return [f0, f1, f2, f3]


class DepthDecoder(nn.Module):
    """Multi-scale depth decoder with skip connections.

    Produces depth at 4 scales (H/16 → H/2).  Full-res depth is
    obtained by bilinearly upsampling the scale-0 output.
    """

    def __init__(self, num_ch_enc=(64, 64, 128, 256)):
        super().__init__()
        ch = [16, 32, 64, 128]

        self.up3   = ConvBlock(num_ch_enc[3], ch[3])
        self.fuse2 = ConvBlock(ch[3] + num_ch_enc[2], ch[2])
        self.up2   = ConvBlock(ch[2], ch[2])
        self.fuse1 = ConvBlock(ch[2] + num_ch_enc[1], ch[1])
        self.up1   = ConvBlock(ch[1], ch[1])
        self.fuse0 = ConvBlock(ch[1] + num_ch_enc[0], ch[0])

        self.depth_3 = nn.Conv2d(ch[3], 1, 3, padding=1)
        self.depth_2 = nn.Conv2d(ch[2], 1, 3, padding=1)
        self.depth_1 = nn.Conv2d(ch[1], 1, 3, padding=1)
        self.depth_0 = nn.Conv2d(ch[0], 1, 3, padding=1)

        self.min_depth = 0.1
        self.max_depth = 100.0

    def _to_depth(self, x):
        return self.min_depth + (self.max_depth - self.min_depth) * torch.sigmoid(x)

    def forward(self, features):
        f0, f1, f2, f3 = features
        depths = {}

        x = self.up3(f3)
        depths[3] = self._to_depth(self.depth_3(x))

        x = F.interpolate(x, size=f2.shape[2:], mode="nearest")
        x = self.fuse2(torch.cat([x, f2], dim=1))
        depths[2] = self._to_depth(self.depth_2(x))

        x = self.up2(x)
        x = F.interpolate(x, size=f1.shape[2:], mode="nearest")
        x = self.fuse1(torch.cat([x, f1], dim=1))
        depths[1] = self._to_depth(self.depth_1(x))

        x = self.up1(x)
        x = F.interpolate(x, size=f0.shape[2:], mode="nearest")
        x = self.fuse0(torch.cat([x, f0], dim=1))
        depths[0] = self._to_depth(self.depth_0(x))

        return depths, x  # x = finest-scale features


class BEVLifter(nn.Module):
    """LSS-style Lift-Splat for a single forward-facing camera.

    1. Predict a discrete depth distribution per pixel (at H/16 res)
    2. Outer-product with learned context features
    3. Scatter-add into a BEV voxel grid
    4. Refine with a small CNN
    """

    def __init__(self, in_channels=256, feat_channels=64,
                 x_range=(-10, 10), z_range=(0.5, 50),
                 resolution=0.5, depth_bins=48,
                 depth_min=0.5, depth_max=50.0):
        super().__init__()
        self.x_range = x_range
        self.z_range = z_range
        self.resolution = resolution
        self.nx = int((x_range[1] - x_range[0]) / resolution)
        self.nz = int((z_range[1] - z_range[0]) / resolution)
        self.depth_bins = depth_bins
        self.feat_channels = feat_channels

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        self.depth_pred = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, depth_bins, 1),
        )
        self.bev_encoder = nn.Sequential(
            ConvBlock(feat_channels, feat_channels),
            ConvBlock(feat_channels, feat_channels),
        )
        self.register_buffer(
            "depth_centers",
            torch.linspace(depth_min, depth_max, depth_bins),
        )

    def forward(self, encoder_features, K, img_shape):
        """
        Args:
            encoder_features: (B, 256, Hf, Wf) encoder scale-3 features
            K:                (B, 3, 3) camera intrinsics (full-res)
            img_shape:        (H, W) original image size
        Returns:
            bev: (B, feat_channels, nz, nx) BEV feature map
        """
        B, _, Hf, Wf = encoder_features.shape
        H, W = img_shape
        D, C = self.depth_bins, self.feat_channels

        context = self.reduce(encoder_features)
        depth_logits = self.depth_pred(context)
        depth_probs = F.softmax(depth_logits, dim=1)

        # Scale intrinsics to feature resolution
        K_s = K.clone()
        K_s[:, 0, :] *= (Wf / W)
        K_s[:, 1, :] *= (Hf / H)
        K_inv = torch.inverse(K_s)

        # Pixel grid → camera rays at feature resolution
        v, u = torch.meshgrid(
            torch.arange(Hf, dtype=torch.float32, device=K.device),
            torch.arange(Wf, dtype=torch.float32, device=K.device),
            indexing="ij",
        )
        pix = torch.stack([u, v, torch.ones_like(u)], dim=0).reshape(3, -1)
        pix = pix.unsqueeze(0).expand(B, -1, -1)
        rays = K_inv @ pix                          # (B, 3, N)
        N = Hf * Wf

        # 3D frustum → BEV grid indices  (x and z only)
        dc = self.depth_centers.view(1, D, 1)       # (1, D, 1)
        x_3d = rays[:, 0:1, :] * dc                 # (B, D, N)
        z_3d = rays[:, 2:3, :] * dc                 # (B, D, N)

        bx = ((x_3d - self.x_range[0]) / self.resolution).long()
        bz = ((z_3d - self.z_range[0]) / self.resolution).long()
        valid = (bx >= 0) & (bx < self.nx) & (bz >= 0) & (bz < self.nz)

        # Weighted features = context × depth_probs
        ctx_flat  = context.reshape(B, C, N)         # (B, C, N)
        prob_flat = depth_probs.reshape(B, D, N)     # (B, D, N)
        weighted  = ctx_flat.unsqueeze(2) * prob_flat.unsqueeze(1)  # (B, C, D, N)

        # Flatten D×N and scatter into BEV grid
        weighted_flat = weighted.reshape(B, C, D * N)
        bev_idx       = (bz * self.nx + bx).reshape(B, D * N)
        valid_flat    = valid.reshape(B, D * N)

        bev_idx = bev_idx.clamp(0, self.nz * self.nx - 1)
        weighted_flat = weighted_flat * valid_flat.unsqueeze(1).float()

        bev = torch.zeros(B, C, self.nz * self.nx, device=K.device)
        for b in range(B):
            bev[b].scatter_add_(
                1,
                bev_idx[b : b + 1].expand(C, -1),
                weighted_flat[b],
            )
        bev = bev.reshape(B, C, self.nz, self.nx)
        bev = self.bev_encoder(bev)
        return bev


# ── Training helpers ─────────────────────────────────────────────────────

def multiscale_photometric_loss(encoder, decoder, img_curr, img_prev, img_next,
                                K_batch, K_inv_batch, T_c2v, T_v2c,
                                pose_curr, pose_prev, pose_next):
    """Compute multi-scale photometric loss across 4 depth scales."""
    B, _, H, W = img_curr.shape

    features = encoder(img_curr)
    depths, finest_feat = decoder(features)

    T_curr_to_prev = compute_relative_transform(pose_curr, pose_prev, T_c2v, T_v2c)
    T_curr_to_next = compute_relative_transform(pose_curr, pose_next, T_c2v, T_v2c)

    total_photo = 0.0
    for scale in range(4):
        depth_s = F.interpolate(depths[scale], size=(H, W),
                                mode="bilinear", align_corners=True)

        wp, mp = warp_image(img_prev, depth_s, K_batch, K_inv_batch, T_curr_to_prev)
        wn, mn = warp_image(img_next, depth_s, K_batch, K_inv_batch, T_curr_to_next)

        l1_p = (wp - img_curr).abs().mean(dim=1, keepdim=True)
        l1_n = (wn - img_curr).abs().mean(dim=1, keepdim=True)
        cmask = torch.max(mp, mn)
        pmap  = torch.min(l1_p, l1_n)
        total_photo += (pmap * cmask).sum() / cmask.sum().clamp(min=1) / 4.0

    # Full-res depth for smoothness
    depth_full = F.interpolate(depths[0], size=(H, W),
                               mode="bilinear", align_corners=True)

    return total_photo, depth_full, features


def train_one_epoch(encoder, decoder, bev_lifter, dataloader, optimizer,
                    device, T_c2v, T_v2c, smooth_weight=0.001):
    """Train one epoch with multi-scale photometric loss + BEV forward."""
    encoder.train()
    decoder.train()
    bev_lifter.train()
    total_loss = 0.0
    count = 0

    for batch in dataloader:
        img_prev  = batch["img_prev"].to(device)
        img_curr  = batch["img_curr"].to(device)
        img_next  = batch["img_next"].to(device)
        pose_prev = batch["pose_prev"].to(device)
        pose_curr = batch["pose_curr"].to(device)
        pose_next = batch["pose_next"].to(device)
        K_batch   = batch["K"].to(device)
        K_inv_batch = batch["K_inv"].to(device)

        B, _, H, W = img_curr.shape

        # Multi-scale photometric loss
        photo_loss, depth_full, features = multiscale_photometric_loss(
            encoder, decoder, img_curr, img_prev, img_next,
            K_batch, K_inv_batch, T_c2v, T_v2c,
            pose_curr, pose_prev, pose_next,
        )

        # BEV forward pass (gradients flow through depth distribution)
        bev = bev_lifter(features[3], K_batch, (H, W))

        # Smoothness regularization
        mean_d = depth_full.mean().detach().clamp(min=1e-3)
        smooth = smoothness_loss(depth_full / mean_d, img_curr)

        loss = photo_loss + smooth_weight * smooth

        optimizer.zero_grad()
        loss.backward()
        all_params = (list(encoder.parameters())
                      + list(decoder.parameters())
                      + list(bev_lifter.parameters()))
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


# ── Instantiate model and optimizer ──────────────────────────────────────

encoder    = ResNetEncoder(pretrained=True).to(DEVICE)
decoder    = DepthDecoder(encoder.num_ch_enc).to(DEVICE)
bev_lifter = BEVLifter(in_channels=256, feat_channels=64).to(DEVICE)

all_params = (list(encoder.parameters())
              + list(decoder.parameters())
              + list(bev_lifter.parameters()))
optimizer = torch.optim.Adam(all_params, lr=LEARNING_RATE)

num_params = sum(p.numel() for p in all_params)
print(f"  ResNetEncoder + DepthDecoder + BEVLifter ({num_params:,} parameters)")
print(f"  Optimizer: Adam, lr={LEARNING_RATE}")
print(f"  Device: {DEVICE}")

# ── Training loop ────────────────────────────────────────────────────────
print(f"\n  Starting training for {NUM_EPOCHS} epochs...")
losses = []

for epoch in range(NUM_EPOCHS):
    epoch_loss = train_one_epoch(
        encoder, decoder, bev_lifter, dataloader, optimizer, DEVICE,
        T_cam_to_vehicle_torch, T_vehicle_to_cam_torch,
    )
    losses.append(epoch_loss)
    print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} — loss: {epoch_loss:.6f}")

# ── Plot loss curve ──────────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(range(1, len(losses) + 1), losses, "b-o", linewidth=2, markersize=4)
plt.xlabel("Epoch")
plt.ylabel("Photometric Loss")
plt.title("Self-Supervised Training — ResNet18 + Multi-Scale Depth + BEV")
plt.grid(True, alpha=0.3)
plt.tight_layout()
loss_plot_path = DEBUG_DIR / "training_loss.png"
plt.savefig(loss_plot_path, dpi=100)
plt.close()
print(f"\n  ✓ Loss curve saved: {loss_plot_path}")

# ── Final visualization ─────────────────────────────────────────────────
print("\n  Generating final visualization...")
encoder.eval()
decoder.eval()
bev_lifter.eval()

with torch.no_grad():
    sample = dataset[len(dataset) // 2]
    img_curr = sample["img_curr"].unsqueeze(0).to(DEVICE)
    img_prev = sample["img_prev"].unsqueeze(0).to(DEVICE)
    depth_gt = sample["depth"].unsqueeze(0).to(DEVICE)
    K_b      = sample["K"].unsqueeze(0).to(DEVICE)
    K_inv_b  = sample["K_inv"].unsqueeze(0).to(DEVICE)
    pose_prev_b = sample["pose_prev"].unsqueeze(0).to(DEVICE)
    pose_curr_b = sample["pose_curr"].unsqueeze(0).to(DEVICE)

    B, _, H, W = img_curr.shape

    features = encoder(img_curr)
    depths, _ = decoder(features)
    bev = bev_lifter(features[3], K_b, (H, W))

    depth_pred = F.interpolate(depths[0], size=(H, W),
                               mode="bilinear", align_corners=True)

    T_rel = compute_relative_transform(
        pose_curr_b, pose_prev_b,
        T_cam_to_vehicle_torch, T_vehicle_to_cam_torch,
    )
    warped, mask = warp_image(img_prev, depth_pred, K_b, K_inv_b, T_rel)

    fig, axes = plt.subplots(2, 3, figsize=(24, 10))

    axes[0, 0].imshow(img_curr[0].cpu().permute(1, 2, 0).numpy())
    axes[0, 0].set_title("Current frame", fontsize=14)

    axes[0, 1].imshow(depth_pred[0, 0].cpu().numpy(), cmap="turbo", vmin=0, vmax=30)
    axes[0, 1].set_title("Predicted depth (multi-scale)", fontsize=14)

    axes[0, 2].imshow(warped[0].cpu().permute(1, 2, 0).numpy().clip(0, 1))
    axes[0, 2].set_title("Warped prev → current", fontsize=14)

    # BEV visualization
    bev_vis = bev[0].mean(dim=0).cpu().numpy()  # average over channels
    axes[1, 0].imshow(bev_vis, cmap="viridis", origin="lower",
                      extent=[*bev_lifter.x_range, *bev_lifter.z_range],
                      aspect="auto")
    axes[1, 0].set_xlabel("X (m)")
    axes[1, 0].set_ylabel("Z (m, forward)")
    axes[1, 0].set_title("BEV feature map (top-down)", fontsize=14)

    if depth_gt is not None:
        axes[1, 1].imshow(depth_gt[0, 0].cpu().numpy(), cmap="turbo", vmin=0, vmax=30)
        axes[1, 1].set_title("Stereo depth (reference)", fontsize=14)

    diff = (warped[0] - img_curr[0]).abs().mean(dim=0).cpu().numpy()
    axes[1, 2].imshow(diff, cmap="hot", vmin=0, vmax=0.3)
    axes[1, 2].set_title("Photometric error", fontsize=14)

    for ax in axes.flat:
        ax.axis("off")
    axes[1, 0].axis("on")  # keep axis labels for BEV

    plt.suptitle("ResNet-18 + Multi-Scale Depth + BEV — Results",
                 fontsize=16, y=1.01)
    plt.tight_layout()

    final_path = DEBUG_DIR / "final_results.png"
    plt.savefig(final_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Final results saved: {final_path}")
    print(f"    Predicted depth range: {depth_pred.min().item():.2f} – "
          f"{depth_pred.max().item():.2f} m")
    print(f"    BEV shape: {bev.shape}")

# ── Save model checkpoint ────────────────────────────────────────────────
checkpoint_path = BASE_DIR / "pipeline_checkpoint.pth"
torch.save({
    "encoder_state_dict": encoder.state_dict(),
    "decoder_state_dict": decoder.state_dict(),
    "bev_lifter_state_dict": bev_lifter.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "losses": losses,
    "epoch": NUM_EPOCHS,
}, checkpoint_path)
print(f"  ✓ Checkpoint saved: {checkpoint_path}")

print("\n" + "=" * 72)
print("  Pipeline complete!")
print(f"  Debug outputs: {DEBUG_DIR}")
print(f"  Checkpoint:    {checkpoint_path}")
print("=" * 72)
