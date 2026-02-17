# Task: Feature Distillation with Foundation Model

- [x] Research availability of DINOv2/DINOv3 semantic segmentation models <!-- id: 0 -->
- [x] Create implementation plan <!-- id: 1 -->
- [x] Implement semantic segmentation script <!-- id: 2 -->
    - [x] Load images from `data/images` <!-- id: 3 -->
    - [x] Apply foundation model (Mask2Former/DINOv2) <!-- id: 4 -->
    - [x] Generate segmentation masks and labels <!-- id: 5 -->
    - [x] Save output to `data/segmentation` <!-- id: 6 -->
- [x] Verify results on a few images <!-- id: 7 -->

- [x] Create feature distillation implementation plan <!-- id: 8 -->
- [x] Implement feature extraction script (`extract_features.py`) <!-- id: 9 -->
    - [x] Load DINOv2 model (e.g., `facebook/dinov2-large`) <!-- id: 10 -->
    - [x] Extract patch features (dense features) <!-- id: 11 -->
    - [x] Generate PCA visualizations (RGB images of features) <!-- id: 14 -->
    - [x] Save features (e.g., .npy or .pt) to `data/features` <!-- id: 12 -->
- [x] Verify feature extraction output <!-- id: 13 -->

- [ ] Create point cloud generation implementation plan <!-- id: 15 -->
- [ ] Implement point cloud generation script (`generate_pointcloud.py`) <!-- id: 16 -->
    - [ ] Load intrinsics <!-- id: 17 -->
    - [ ] Generate RGB Point Cloud <!-- id: 22 -->
    - [ ] Generate Semantic Point Cloud <!-- id: 23 -->
    - [ ] Generate Feature Point Cloud (PCA-colored) <!-- id: 24 -->
    - [ ] Save PLY files to `data/pointclouds/{rgb,semantic,features}` <!-- id: 19 -->
- [ ] Verify point cloud output <!-- id: 20 -->
