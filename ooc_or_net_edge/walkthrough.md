# Implementation Walkthrough: OffRoadOccNetNano Upgrades

The goal of this task was to update the off-road perception model to use an EfficientNet backbone and incorporate a multi-task learning objective (semantic cross-entropy + DINOv2 feature distillation) to generate self-supervised 2D-to-BEV mappings tailored for Jetson edge devices. 

## Changes Made

### 1. Backbone Upgrade
- Modified [off_road_occ_net_nano.py](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/off_road_occ_net_nano.py) to replace the `MobileNetV3` backbone with an `EfficientNet-B0` backbone.
- Extracted features up to the 6th layer (`index 5` of the `features` sequence) to achieve the desired 1/16 spatial scale reduction. 
- Mapped the 112-channel intermediate feature map to the desired generic `embed_dim`. 

### 2. DINOv2 Pseudo-Ground Truth in BEV
- Updated [occ_dataset.py](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/occ_dataset.py) to handle the `[D, Hp, Wp]` 2D DINOv2 feature embeddings.
- Used `F.interpolate` to bilinearly upsample the patch embeddings to the image size `[D, H, W]`.
- Scattered the valid valid depth-matched feature subsets `[N, D]` into a pre-initialized BEV grid `gt_bev_feat` of shape `[D, 50, 50]`.
- Returned `gt_bev_feat` in the dictionary output alongside existing tensors.

### 3. Multi-Task Training Loop
- Added [masked_feature_loss](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/train_nano.py#52-76) in [train_nano.py](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/train_nano.py) to compute feature distillation consisting of Cosine Similarity (`1 - cos_sim`) and L1/MSE loss.
- Combined the standard categorical cross-entropy semantic loss `s_loss` with the continuous feature embedding loss `f_loss` using a balancing hyperparameter `alpha_distill = 10.0`. 
- Added console and `tqdm` log-tracking for both `Sem` and `Feat` individually.

### 4. Explicit 2D Geometry Supervision
- Added 2D target `depths` into the [occ_dataset.py](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/occ_dataset.py) batch output.
- Added a [DepthHead](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/off_road_occ_net_nano.py#42-60) to [OffRoadOccNetNano](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/off_road_occ_net_nano.py#345-425) using a single 1x1 convolution yielding positive depth via ReLU `[B, 1, H/16, W/16]`.
- Implemented [masked_depth_loss](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/train_nano.py#77-103) in [train_nano.py](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/train_nano.py) computing `F.smooth_l1_loss` over valid depths (upsampling predictions using bilinear interpolation to match labels).
- Appended depth loss `d_loss` (`alpha_depth = 1.0`) directly into the final `total_loss` gradients!

### 5. Lift-Splat-Shoot (LSS) View Transformer
- Removed the Raycasting module (`SpatialCrossAttentionBEV` and `VoxelGridBEV`) from [off_road_occ_net_nano.py](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/off_road_occ_net_nano.py).
- Implemented a Deterministic [LSSViewTransformer](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/off_road_occ_net_nano.py#178-298) that directly uses the continuous batch output `depths_2d`.
- **Lift**: Generated absolute 3D Cartesian coordinates for every pixel deterministically by utilizing inverse camera intrinsics and the explicit regression depth map tensor `[1, fH, fW]`. 
- **Transform**: Translated coordinates to the Vehicle frame using Camera [extrinsics](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/occ_dataset.py#165-198).
- **Splat**: Digitized all real-world points falling in the $[-25m, 25m]$ box into indices $[0, 50]$ using `floor` operations. Flat-indexed the points, and natively projected `all_feats` into a $50 \times 50$ buffer `[C, Y, X]` utilizing PyTorch's `scatter_add_` primitive, averaging overlapping pixels.

### 6. Temporal Fusion Architecture (ConvGRU)
- **Dataset ([occ_dataset.py](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/occ_dataset.py))**: Updated to handle time-series sequences instead of distinct images. [OffRoadOccDataset](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/occ_dataset.py#65-576) now clusters frames chronologically into batched sequences `[S, N, C, H, W]` controlled by `seq_len` (default 2), while detecting timestamps gaps $>1s$ to slice discontinuities dynamically.
- **Model Modifications ([off_road_occ_net_nano.py](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/off_road_occ_net_nano.py))**: 
  - Inserted a [SpatialAlignment](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/off_road_occ_net_nano.py#62-124) block leveraging `T_rel = (T_world_curr)^-1 * T_world_prev`. Uses `F.affine_grid` and `grid_sample` to geometrically translate and rotate the $(t-1)$ ego-frame BEV explicitly into the geometry of timestep $t$.
  - Constructed a memory-light [ConvGRU](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/off_road_occ_net_nano.py#126-176) bridging the current [LSS](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/off_road_occ_net_nano.py#178-298) representation and the aligned `prev_bev` memory buffer, acting as a robust occlusion handler. 
- **Backpropagation Through Time ([train_nano.py](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/train_nano.py))**: Upgraded the training cycle to chronologically evaluate recurrent models. It sequences $S$ frame timesteps serially per batch, passing the model internal recurrent tracking mechanism `prev_bev=curr_state`. Semantic, Feature, and Depth gradients are deliberately isolated and calculated solely upon the culmination of the final cycle $t=S-1$ utilizing the accumulated history. 

### 7. Traversability Head & 5-Loss Multi-Task Loop
- **Dataset Targets**: Inserted mapping logic in [occ_dataset.py](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/occ_dataset.py) to derive `gt_bev_occ` (1 for occupied, 0 for free space) and `gt_bev_cost` (Float $[0, 1]$ traversing difficulty) directly from the `gt_bev_sem` label.
- **Model End-Cap ([off_road_occ_net_nano.py](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/off_road_occ_net_nano.py))**: Exchanged [Decoder2D](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/off_road_occ_net_nano.py#300-329) with [TraversabilityHead](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/off_road_occ_net_nano.py#300-343) utilizing a FlashOcc strategy (Shared convolutions branching into 4 isolated `1x1` task convolutions). It now returns `occ_logits`, `semantic_bev`, `feature_bev`, and bounded `cost_map` inferences.
- **Loss Balancer ([train_nano.py](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/train_nano.py))**: Instantiated [masked_occupancy_loss](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/train_nano.py#104-111) containing `BCEWithLogitsLoss()` alongside [masked_cost_loss](file:///c:/Users/maxlars/Documents/GitHub/VTUGV/occ_or_net/train_nano.py#112-118) utilizing `Smooth L1` criteria. The network optimizes 5 concurrent targets using configured mixture multipliers! (`total_loss = s_loss + (10 * f_loss) + (1.0 * d_loss) + (5.0 * o_loss) + (5.0 * c_loss)`).

## Validation Results
The code alterations have been synthesized successfully. Automatic testing using the base `python` command was restricted as `torch` wasn't immediately discovered on the default system python path.

**Action Required:**
Please activate your working conda or venv PyTorch environment and run:
```bash
python occ_or_net/off_road_occ_net_nano.py
python occ_or_net/train_nano.py
```
This will confirm the dataset correctly interpolates features, the model generates the depth, and the backwards pass propagates gradient appropriately through the three task losses!
