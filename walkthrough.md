# Self-Supervised 4D Projection Pipeline — Walkthrough

## What Was Built

[pipeline_4d.py](file:///c:/Users/maxlars/UGV_research_pipeline/pipeline_4d.py) — a complete **self-supervised 4D occupancy projection pipeline** (~580 lines) structured as 6 clearly-labeled sections that match the [new_notebook.ipynb](file:///c:/Users/maxlars/UGV_research_pipeline/new_notebook.ipynb) headers. Each section can be copied directly into a notebook cell.

## Pipeline Architecture

```mermaid
graph LR
    A["RGB Frame T"] --> B["Stereo Depth"]
    B --> C["Backproject to 3D<br/>D · K⁻¹ · [u,v,1]"]
    C --> D["Transform via Pose<br/>T_v2w · T_c2v"]
    D --> E["Reproject to 2D<br/>K · P_3d"]
    E --> F["Grid Sample<br/>Warp neighbor"]
    F --> G["Photometric Loss<br/>SSIM + L1"]
    G --> H["Depth Refiner<br/>Network Update"]
    H --> B
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **TartanVO over raw IMU** | Provides direct 6-DOF poses; IMU quaternions are all zeros |
| **Nearest-timestamp matching** | Mean offset ~50ms between image and odometry timestamps |
| **Min photometric loss** | `min(loss_prev, loss_next)` handles occlusions at frame boundaries |
| **Softplus depth output** | Guarantees positive depth predictions from the network |
| **Log-depth input normalization** | Compresses the large depth range (0.5–100m) for network input |

## Training Results

10 epochs on the 77-frame subset (CPU, ~15 min total):

| Epoch | Loss |
|---|---|
| 1 | 0.3955 |
| 5 | 0.3520 |
| 10 | ~0.347 |

![Training loss curve](C:\Users\maxlars\.gemini\antigravity\brain\87ad7ed6-457a-4357-bdb9-349857e36fdb\training_loss.png)

## Projection Sanity Check

The pipeline correctly warps a neighboring frame into the current viewpoint using stereo depth + TartanVO odometry:

![Projection sanity check — target vs source vs warped vs mask](C:\Users\maxlars\.gemini\antigravity\brain\87ad7ed6-457a-4357-bdb9-349857e36fdb\projection_sanity_check.png)

## Final Results — Before/After Depth Refinement

![Comparison of initial stereo depth vs refined depth warping](C:\Users\maxlars\.gemini\antigravity\brain\87ad7ed6-457a-4357-bdb9-349857e36fdb\final_results.png)

## Output Files

| File | Description |
|---|---|
| [pipeline_4d.py](file:///c:/Users/maxlars/UGV_research_pipeline/pipeline_4d.py) | Complete pipeline script |
| [extracted/projection_debug/projection_sanity_check.png](file:///c:/Users/maxlars/UGV_research_pipeline/extracted/projection_debug/projection_sanity_check.png) | Projection verification |
| [extracted/projection_debug/training_loss.png](file:///c:/Users/maxlars/UGV_research_pipeline/extracted/projection_debug/training_loss.png) | Loss curve |
| [extracted/projection_debug/final_results.png](file:///c:/Users/maxlars/UGV_research_pipeline/extracted/projection_debug/final_results.png) | Before/after depth refinement |
| [depth_refiner_checkpoint.pth](file:///c:/Users/maxlars/UGV_research_pipeline/depth_refiner_checkpoint.pth) | Trained model weights |

## Next Steps

1. **Copy into notebook** — each `SECTION` block in [pipeline_4d.py](file:///c:/Users/maxlars/UGV_research_pipeline/pipeline_4d.py) maps to a cell
2. **Swap in QueryOcc** — replace [SimpleDepthRefiner](file:///c:/Users/maxlars/UGV_research_pipeline/pipeline_4d.py#775-841) with a voxel-query architecture
3. **Scale data** — load more TartanDrive 2.0 sequences
4. **Add augmentation** — color jitter, random crops for robustness
5. **GPU training** — enable CUDA for realistic epoch times
