# Walkthrough: Depth-Free Self-Supervised Pipeline

## What Changed

Removed all depth file dependencies from [new_notebook.ipynb](file:///Users/maxmagnusson/Documents/Master%20Thesis%20UGV%20Route%20Planning/Code/VTUGV/new_notebook.ipynb) so the pipeline is **fully self-supervised** — the `DepthNet` predicts depth from RGB alone, trained via photometric consistency loss.

### Modified Files

| File | Change |
|------|--------|
| [new_notebook.ipynb](file:///Users/maxmagnusson/Documents/Master%20Thesis%20UGV%20Route%20Planning/Code/VTUGV/new_notebook.ipynb) | 9 cells patched (see details below) |
| [pipeline_overview.md](file:///Users/maxmagnusson/Documents/Master%20Thesis%20UGV%20Route%20Planning/Code/VTUGV/pipeline_overview.md) | Updated references: stereo depth → predicted, SimpleDepthRefiner → DepthNet |
| [patch_notebook.py](file:///Users/maxmagnusson/Documents/Master%20Thesis%20UGV%20Route%20Planning/Code/VTUGV/patch_notebook.py) | One-time migration script (can be deleted after verifying) |

### Key Cell Changes

1. **Config** — Removed `DEPTH_DIR` path and its print statement
2. **`TartanDrive4DDataset`** — Removed `depth_dir` param, removed depth file existence check (was filtering out all 104 frames → 0 triplets), removed depth loading in `__getitem__`
3. **Dataset instantiation** — Removed `depth_dir=DEPTH_DIR` argument
4. **`SimpleDepthRefiner` → `DepthNet`** — Changed from 4-channel (RGB+depth) to 3-channel (RGB) input. Renamed class. `forward(rgb, depth_init)` → `forward(rgb)`
5. **`train_one_epoch`** — Removed `batch["depth"]` loading and depth preprocessing. Calls `model(img_curr)` instead of `model(img_curr, depth_input)`
6. **Model instantiation** — `DepthNet()` instead of `SimpleDepthRefiner(in_channels=4)`
7. **Sanity check** — Uses `model(img_tgt)` for depth prediction instead of loading from file
8. **Final visualization** — Shows predicted depth only (no stereo comparison)
9. **Checkpoint** — Filename: `depth_net_checkpoint.pth`

## Verification

- ✅ Notebook JSON is valid (parses correctly)
- ✅ No remaining references to `DEPTH_DIR`, `depth_init`, `SimpleDepthRefiner`, or `batch["depth"]`
- ⬜ **User should run notebook end-to-end** — expect ~102 valid triplets from 104 frames

## Expected Output When Run

```
── Section 3: Dataset & DataLoader ──
  TartanDrive4DDataset: ~102 valid triplets from 104 frames
  Sample shapes:
    img_curr:  torch.Size([3, 544, 1024])
    ...
```
