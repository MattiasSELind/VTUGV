"""
Dense metric depth from DepthAnythingV2 + sparse LiDAR alignment.

Pipeline:
  1. Run DepthAnythingV2-Small on each left camera image → relative depth map.
  2. Project the closest LiDAR scan into the image (timestamp-matched).
  3. Least-squares fit  metric_depth = scale * relative_depth + offset
     using overlapping LiDAR points as ground truth.
  4. Save 16-bit PNG (1 unit = 1 mm) and a colourmap visualisation.

Usage:  python generate_depth.py
"""

import os
import json
import numpy as np
import cv2
import torch
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_DIR        = Path(r"C:\Users\matli\Desktop\Dataset Preprocessing\extracted\downsampled_stereo_left_images")
PC_DIR         = Path(r"C:\Users\matli\Desktop\Dataset Preprocessing\extracted\pointclouds_center")
CAM_INFO_DIR   = Path(r"C:\Users\matli\Desktop\Dataset Preprocessing\extracted\stereo_left_camera_info")
IMU_CSV        = Path(r"C:\Users\matli\Desktop\Dataset Preprocessing\extracted\imu_ouster\imu.csv")
OUT_DIR        = Path(r"C:\Users\matli\Desktop\Dataset Preprocessing\extracted\depth_maps")

MODEL_NAME     = "depth-anything/Depth-Anything-V2-Small-hf"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

MIN_DEPTH      = 0.5   # metres – ignore closer LiDAR points
MAX_DEPTH      = 30.0  # metres – clip far
MIN_LIDAR_PTS  = 50    # minimum projected LiDAR points to attempt alignment
# ──────────────────────────────────────────────────────────────────────────────

# ── Camera intrinsics (adjusted for 1280×720) ────────────────────────────────
K = np.array([
    [897.929,   0.0,   648.650],
    [  0.0,   974.238, 256.277],
    [  0.0,     0.0,     1.0  ],
], dtype=np.float64)

# ── LiDAR → Camera extrinsic (from transformslid2left_cam.yaml) ──────────────
T_cam_lidar = np.array([
    [ 0.0, -1.0,  0.0, -0.2   ],
    [ 0.0,  0.0, -1.0, -0.275 ],
    [ 1.0,  0.0,  0.0, -0.355 ],
    [ 0.0,  0.0,  0.0,  1.0   ],
], dtype=np.float64)


# ── Build timestamp index ────────────────────────────────────────────────────
def build_timestamp_index():
    """Return camera timestamps (from cam_info jsons) and LiDAR base time."""
    # Camera timestamps
    cam_info_files = sorted(f for f in os.listdir(CAM_INFO_DIR) if f.endswith(".json"))
    cam_timestamps = []
    for cf in cam_info_files:
        with open(os.path.join(CAM_INFO_DIR, cf)) as f:
            cam_timestamps.append(json.load(f)["timestamp"])
    cam_timestamps = np.array(cam_timestamps)

    # LiDAR base time from first IMU entry (same clock as Ouster LiDAR)
    with open(IMU_CSV, "r") as f:
        f.readline()  # skip header
        imu_t0 = float(f.readline().split(",")[0])

    # Map cam image filenames to their timestamps
    cam_fname_to_ts = {}
    for cf in cam_info_files:
        fname = cf.replace(".json", ".png")
        with open(os.path.join(CAM_INFO_DIR, cf)) as f:
            cam_fname_to_ts[fname] = json.load(f)["timestamp"]

    return cam_timestamps, cam_info_files, imu_t0, cam_fname_to_ts


def find_closest_lidar(cam_ts, imu_t0, num_lidar_scans):
    """Find the closest LiDAR scan index for a given camera timestamp."""
    lidar_idx = round((cam_ts - imu_t0) / 0.1)  # 10 Hz LiDAR
    lidar_idx = max(0, min(lidar_idx, num_lidar_scans - 1))
    return lidar_idx


# ── LiDAR projection ─────────────────────────────────────────────────────────
def project_lidar(pc_path, H, W):
    """
    Load a LiDAR scan and project into image.
    Returns: u, v, depth arrays for valid projected points.
    """
    points = np.load(pc_path, allow_pickle=True)
    x, y, z = points['x'], points['y'], points['z']
    rng = points['range']

    # Filter invalid points
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
    xyz = np.stack([x[valid], y[valid], z[valid]], axis=0)  # (3, N)

    # Homogeneous → camera frame
    ones = np.ones((1, xyz.shape[1]))
    xyz_h = np.vstack([xyz, ones])                          # (4, N)
    xyz_cam = (T_cam_lidar @ xyz_h)[:3]                     # (3, N)

    # Keep points in front of camera and within depth range
    mask_front = (xyz_cam[2] > MIN_DEPTH) & (xyz_cam[2] < MAX_DEPTH)
    xyz_cam = xyz_cam[:, mask_front]

    # Project → pixel coordinates
    uvw = K @ xyz_cam
    depth = uvw[2]
    u = (uvw[0] / depth).astype(int)
    v = (uvw[1] / depth).astype(int)

    # Keep points inside image
    mask_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[mask_img], v[mask_img], depth[mask_img]


# ── Alignment ─────────────────────────────────────────────────────────────────
def align_depth(relative_depth, u_lidar, v_lidar, depth_lidar):
    """
    Align in inverse-depth (disparity) space.

    DepthAnythingV2 outputs values proportional to 1/depth.  A linear fit
    in metric space saturates at ~15 m because far pixels compress into a
    narrow band.  Instead we fit:

        1/depth_metric = scale * (1/d_rel) + offset

    ⇒  depth_metric = 1 / (scale / d_rel + offset)

    Returns (scale, offset, rmse_m).
    """
    d_rel = relative_depth[v_lidar, u_lidar].astype(np.float64)

    # Guard against zeros
    valid = (d_rel > 1e-6) & (depth_lidar > 1e-6)
    d_rel = d_rel[valid]
    d_lid = depth_lidar[valid]

    inv_rel = 1.0 / d_rel
    inv_lid = 1.0 / d_lid      # target: inverse metric depth

    # Least-squares:  inv_lid = scale * inv_rel + offset
    A = np.stack([inv_rel, np.ones_like(inv_rel)], axis=1)
    result = np.linalg.lstsq(A, inv_lid, rcond=None)
    scale, offset = result[0]

    # RMSE in metric depth
    fitted_inv = scale * inv_rel + offset
    fitted_metric = np.where(fitted_inv > 1e-8, 1.0 / fitted_inv, 0.0)
    rmse = np.sqrt(np.mean((fitted_metric - d_lid) ** 2))
    return scale, offset, rmse


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "vis").mkdir(exist_ok=True)

    # Load model
    print(f"Loading DepthAnythingV2 ({MODEL_NAME}) on {DEVICE}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print("Model loaded.")

    # Timestamp matching setup
    cam_timestamps, cam_info_files, imu_t0, cam_fname_to_ts = build_timestamp_index()
    lidar_files = sorted(PC_DIR.glob("*.npy"))
    num_lidar = len(lidar_files)
    print(f"LiDAR scans: {num_lidar}")

    # Process all images
    img_files = sorted(IMG_DIR.glob("*.png"))
    total = len(img_files)
    print(f"Found {total} images\n")

    skipped = 0
    for i, img_path in enumerate(img_files):
        # ── 1. Run DepthAnythingV2 ──
        pil_img = Image.open(img_path).convert("RGB")
        W_img, H_img = pil_img.size

        inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth  # (1, h, w)

        # Resize to original image size
        relative_depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(0),
            size=(H_img, W_img),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()  # (H, W) float32

        # ── 2. Find & project matching LiDAR scan ──
        cam_ts = cam_fname_to_ts.get(img_path.name)
        if cam_ts is None:
            # Fallback: use index-based matching
            idx = i
            lidar_idx = idx // 3  # ~30Hz cam / 10Hz lidar
        else:
            lidar_idx = find_closest_lidar(cam_ts, imu_t0, num_lidar)

        pc_path = PC_DIR / f"{lidar_idx:06d}.npy"
        if not pc_path.exists():
            print(f"  [{i+1}/{total}] {img_path.name} — no LiDAR scan {pc_path.name}, skipping alignment")
            skipped += 1
            continue

        u_lid, v_lid, d_lid = project_lidar(str(pc_path), H_img, W_img)

        if len(u_lid) < MIN_LIDAR_PTS:
            print(f"  [{i+1}/{total}] {img_path.name} — only {len(u_lid)} LiDAR pts, skipping")
            skipped += 1
            continue

        # ── 3. Align relative → metric (in inverse-depth space) ──
        scale, offset, rmse = align_depth(relative_depth, u_lid, v_lid, d_lid)
        # depth_metric = 1 / (scale / d_rel + offset)
        inv_pred = scale / np.maximum(relative_depth, 1e-6) + offset
        metric_depth = np.where(inv_pred > 1e-8, 1.0 / inv_pred, 0.0)
        metric_depth = np.clip(metric_depth, 0, MAX_DEPTH)

        # ── 4. Save ──
        depth_mm = (metric_depth * 1000).astype(np.uint16)
        cv2.imwrite(str(OUT_DIR / img_path.name), depth_mm)

        # Colourmap visualisation
        depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colour = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        depth_colour[depth_mm == 0] = 0
        cv2.imwrite(str(OUT_DIR / "vis" / img_path.name), depth_colour)

        if (i + 1) % 50 == 0 or i == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] {img_path.name}  scale={scale:.4f}  offset={offset:.4f}  RMSE={rmse:.3f}m  LiDAR pts={len(u_lid)}")

    print(f"\nDone! {total - skipped}/{total} frames processed.")
    print(f"Depth maps → {OUT_DIR}")
    print(f"Visualisations → {OUT_DIR / 'vis'}")


if __name__ == "__main__":
    main()