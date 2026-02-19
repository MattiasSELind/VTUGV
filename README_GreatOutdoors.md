
# Great Outdoors Point Cloud Pipeline

This pipeline allows you to generate 3D point clouds from the Great Outdoors dataset (3 cameras) using monocular depth estimation.

## Prerequisites
1.  **Data**: ensure the dataset is downloaded to `~/Downloads/GreatOutdoors`.
    - Expected structure:
        ```
        GreatOutdoors/
        ├── cam1/
        │   ├── image_001.png
        │   └── ...
        ├── cam2/
        │   └── ...
        └── cam3/
            └── ...
        ```
2.  **GPU**: A GPU is highly recommended for `estimate_depth.py`.

## Step 1: Estimate Depth
Run the depth estimation script to generate depth maps for all cameras.
```bash
python estimate_depth.py
```
- **Input**: `~/Downloads/GreatOutdoors`
- **Output**: `~/Downloads/GreatOutdoors_depth`
- **Model**: Uses `Depth-Anything-V2-Small` (downloaded automatically from Hugging Face).

## Step 2: Fuse Point Clouds
Run the fusion script to combine images and depth into point clouds.
```bash
python fuse_outdoors.py
```
- **Input**: Images from `GreatOutdoors` and Depth from `GreatOutdoors_depth`.
- **Output**: `~/Downloads/GreatOutdoors_pointclouds` (PLY files).

## Configuration
- **Intrinsics**: The script uses placeholder intrinsics in `fuse_outdoors.py`. **You must update the `INTRINSICS` dictionary** with the actual values for the Great Outdoors cameras if you want geometrically accurate point clouds.
- **Extrinsics**: The script allows for camera-to-vehicle transforms. Update `EXTRINSICS` in `fuse_outdoors.py` to align the 3 cameras correctly relative to the vehicle (e.g., rotations and translations).

## Output
The output PLY files can be viewed in MeshLab, CloudCompare, or Open3D.
