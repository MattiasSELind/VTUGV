# Multi-View Point Cloud Fusion Implementation Plan

## Goal Description
Extend the current pipeline to support a **multi-view dataset** (e.g., Front + Back cameras). This upgrades the system from a single-camera "2.5D" approach to a **Surround-View** capability, significantly improving the completeness and resolution of the fused point cloud map.

## Significant Changes
> [!IMPORTANT]
> **Depth Estimation Required**: Unlike the previous dataset, the new data likely does not come with pre-computed depth maps. We must integrate a Monocular Depth Estimation model (e.g., **Depth Anything V2** or **ZoeDepth**) to generate depth for each camera view.

## Proposed Changes

### 1. Depth Estimation Integration (`estimate_depth.py`)
- **Action**: Create a new script to generate depth maps for *all* input images.
- **Model**: Use **Depth Anything V2** (Small/Base) for high-quality, metric-relative depth.
- **Output**: Save as [.npy](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/data/depth/depth_000087.npy) in `data/depth/{camera_name}/`.

### 2. Multi-View Processing Updates
- **[segment_images.py](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/segment_images.py)** & **[extract_features.py](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/extract_features.py)**:
    - Modify to iterate over **all camera subdirectories** (e.g., `data/images/front`, `data/images/back`) instead of a single hardcoded path.
    - Output to corresponding `data/{task}/{camera_name}` directories.

### 3. Intrinsics & Extrinsics Management
- **Intrinsics**: Need separate intrinsics ($K_{front}$, $K_{back}$) for each camera.
- **Extrinsics**: Need transforms for each camera relative to the vehicle ($T_{front \to vehicle}$, $T_{back \to vehicle}$).
- **Plan**: Update `data/calibration/` to support multiple config files or a unified `rig.yaml`.

### 4. Upgrade Fusion Script (`fuse_pointclouds.py`)
- **Logic**:
    - Iterate through synchronized timestamps.
    - For each timestamp, load **Front** data AND **Back** data.
    - Generate $P_{front}$ (using $K_{front}, D_{front}$).
    - Generate $P_{back}$ (using $K_{back}, D_{back}$).
    - Transform both to Vehicle frame:
        - $P_{veh} = T_{front \to veh} \times P_{front}$
        - $P_{veh} = T_{back \to veh} \times P_{back}$
    - Transform to World using Odom.
    - Fuse into global map.

## Verification Plan
1.  **Depth Check**: Visualize generated depth maps (since we don't have ground truth).
2.  **Overlap Check**: Verify that "Front" and "Back" point clouds align correctly in the fused map (e.g., ground plane is flat, objects don't "ghost").
