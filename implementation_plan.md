# Self-Supervised 4D Occupancy Projection Pipeline

Build a complete self-supervised training pipeline inside [new_notebook.ipynb](file:///c:/Users/maxlars/UGV_research_pipeline/new_notebook.ipynb) that back-projects RGB frames into 3D+time (4D) using stereo depth and TartanVO odometry, then trains a QueryOcc-style network via photometric consistency — all without manual labels.

## Data Inventory

| Data | Source | Count |
|---|---|---|
| Color frames | `extracted/images/multisense_left_image_rect_color/` | 77 frames (1024×544) |
| Stereo depth | `extracted/depth/depth_*.npy` | 80 maps (float32, meters) |
| Image timestamps | `timestamps.csv` per camera | frame ↔ ROS timestamp |
| TartanVO odometry | `extracted/odometry/tartanvo_odom.csv` | 67 poses (pos xyz + quat xyzw) |
| Camera intrinsics | `extracted/calibration/multisense_intrinsics.txt` | K, P, D, R matrices |
| Extrinsics | `extracted/calibration/extrinsics.yaml` | vehicle↔camera transforms |

> [!IMPORTANT]
> **TartanVO odometry** provides direct 6-DOF poses (position + quaternion), so we use these instead of double-integrating raw IMU (which drifts rapidly). The IMU data can serve as a secondary reference.

## Proposed Changes

Since the user cannot edit `.ipynb` files directly via tool, I will create a **standalone Python script** `pipeline_4d.py` that mirrors the notebook sections. The user can then copy cells into the notebook, or run directly.

### Pipeline Script

#### [NEW] [pipeline_4d.py](file:///c:/Users/maxlars/UGV_research_pipeline/pipeline_4d.py)

Implements all 6 notebook sections as clearly delimited sections:

**Section 0 — Imports**: torch, numpy, cv2, scipy.spatial.transform, matplotlib

**Section 1 — Configuration**:
- Paths to images, depth, odometry, calibration
- Voxel grid parameters (X/Y/Z range, resolution)
- Training hyperparameters (lr, epochs, batch size)

**Section 2 — Data Calibration**:
- Parse `multisense_intrinsics.txt` → K matrix (3×3), projection P (3×4)
- Parse `extrinsics.yaml` → T_vehicle_to_camera (4×4)
- Parse `tartanvo_odom.csv` → per-frame T_vehicle_to_world (4×4)
- Compose full chain: `T_cam_to_world = T_v2w @ inv(T_c2v)`
- Match odometry poses to image timestamps via nearest-timestamp lookup

**Section 3 — Dataset & DataLoader**:
- `TartanDrive4DDataset(torch.utils.data.Dataset)`:
  - Returns triplets: `(img_prev, img_curr, img_next, depth_curr, T_prev, T_curr, T_next, K)`
  - Handles timestamp matching between images and odometry
  - Normalizes images to [0,1]

**Section 4 — Projection (the core 4D pipeline)**:
- `backproject_depth_to_3d(depth, K_inv)` → 3D point cloud in camera frame
- `transform_points(pts_3d, T_src_to_tgt)` → transform to another frame
- `project_3d_to_2d(pts_3d, K)` → project back to pixel coordinates
- `warp_image(src_img, depth_tgt, K, T_tgt_to_src)` → full warping pipeline
- Compose relative transform: `T_rel = T_tgt_inv @ T_src`
- Grid sample to synthesize view from neighboring frame

**Section 5 — Loss Functions & Training Loop**:
- `ssim_loss(pred, target)` — structural similarity (window-based)
- `photometric_loss(pred, target)` — weighted combination: `0.85 * SSIM + 0.15 * L1`
- `smoothness_loss(depth, image)` — edge-aware depth smoothness
- Training loop with Adam optimizer
- Visualization of warped images and loss curves

## Verification Plan

### Automated / Script-Based
1. **Run the script end-to-end**: `python pipeline_4d.py` — should load data, compute projections, and run at least 1 epoch
2. **Sanity check outputs**: The script will save a warped image visualization to `extracted/projection_debug/` showing the reprojected frame alongside the target frame

### Manual Verification (for user)
1. Open the generated visualization images in `extracted/projection_debug/` — warped frames should look roughly like the target frame (not garbled or empty)
2. Copy sections from `pipeline_4d.py` into `new_notebook.ipynb` cells to run interactively
