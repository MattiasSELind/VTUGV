"""
train_bev.py
Training loop for BEV semantic + geometric prediction.

Input:  RGB image (360×640)
Output: BEV grid 100×100 with semantics (13 classes), height (m), slope (°)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, avoids tkinter threading errors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from bev_model import BEVPredictionNet
from bev_dataset import BEVDataset

# ── Configuration ────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
DATA_ROOT = os.path.join(HOME, "Downloads", "Sample Dataset With Semantic Annotations")

BATCH_SIZE    = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS    = 30
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS   = 4
IMG_SIZE      = (360, 640)
NUM_CLASSES   = 13
EMBED_DIM     = 128
VIS_INTERVAL  = 50     # visualise every N batches
CKPT_DIR      = "checkpoints_bev"
VIS_DIR       = "bev_visualizations"

# Loss weights
ALPHA_HEIGHT = 1.0
ALPHA_SLOPE  = 0.5

# Class names and colours for visualisation
CLASS_NAMES = [
    "dirt road", "grass", "tree", "bush", "rock", "mud",
    "water", "sky", "vehicle", "person", "building", "fence", "terrain",
]
CLASS_COLORS = np.array([
    [160,120, 80],[124,252,  0],[ 34,139, 34],[  0,180,  0],
    [160,160,160],[100, 70, 40],[  0,100,255],[135,206,250],
    [  0,  0,180],[255,  0,  0],[ 70, 70, 70],[190,153,153],
    [210,180,140],
], dtype=np.uint8)


# ── Loss functions ───────────────────────────────────────────────────

def masked_semantic_loss(pred, target, mask, class_weights=None):
    """CrossEntropy on valid cells, ignoring 255 (unobserved)."""
    loss = F.cross_entropy(pred, target, weight=class_weights,
                           reduction="none", ignore_index=255)
    masked = loss * mask.float()
    n = mask.sum()
    return masked.sum() / n.clamp(min=1)


def masked_regression_loss(pred, target, mask):
    """Smooth-L1 on valid cells."""
    loss = F.smooth_l1_loss(pred.squeeze(1), target, reduction="none")
    masked = loss * mask.float()
    n = mask.sum()
    return masked.sum() / n.clamp(min=1)


# ── Visualisation ────────────────────────────────────────────────────

def visualise_bev(pred_sem, gt_sem, pred_h, gt_h, pred_s, gt_s, mask, fov,
                  epoch, batch_idx, save_dir=VIS_DIR):
    """Saves a 3×2 comparison plot (GT left, pred right)."""
    os.makedirs(save_dir, exist_ok=True)

    pred_cls = pred_sem.argmax(dim=0).cpu().numpy()
    gt_cls   = gt_sem.cpu().numpy()
    mask_np  = mask.cpu().numpy()
    fov_np   = fov.cpu().numpy()

    def sem_rgb(cls_map, m, f):
        cls_clamped = np.clip(cls_map, 0, 12)
        rgb = CLASS_COLORS[cls_clamped].copy()
        rgb[~f] = [0, 0, 0]
        rgb[f & ~m] = [30, 30, 30]
        return rgb

    fig, axes = plt.subplots(3, 2, figsize=(10, 14))
    fig.suptitle(f"Epoch {epoch}, Batch {batch_idx}", fontsize=13, fontweight="bold")

    # Row 0: Semantics
    axes[0, 0].imshow(sem_rgb(gt_cls, mask_np, fov_np), interpolation="nearest")
    axes[0, 0].set_title("GT Semantics")
    axes[0, 1].imshow(sem_rgb(pred_cls, mask_np, fov_np), interpolation="nearest")
    axes[0, 1].set_title("Pred Semantics")

    # Row 1: Height
    gt_h_np = gt_h.cpu().numpy()
    pred_h_np = pred_h.squeeze(0).cpu().numpy()
    vmin_h, vmax_h = np.nanmin(gt_h_np[mask_np]), np.nanmax(gt_h_np[mask_np]) if mask_np.any() else (0, 1)
    axes[1, 0].imshow(np.where(fov_np, gt_h_np, np.nan), cmap="coolwarm", vmin=vmin_h, vmax=vmax_h)
    axes[1, 0].set_title("GT Height")
    axes[1, 1].imshow(np.where(fov_np, pred_h_np, np.nan), cmap="coolwarm", vmin=vmin_h, vmax=vmax_h)
    axes[1, 1].set_title("Pred Height")

    # Row 2: Slope
    gt_s_np = gt_s.cpu().numpy()
    pred_s_np = pred_s.squeeze(0).cpu().numpy()
    axes[2, 0].imshow(np.where(fov_np, gt_s_np, np.nan), cmap="YlOrRd", vmin=0, vmax=30)
    axes[2, 0].set_title("GT Slope (°)")
    axes[2, 1].imshow(np.where(fov_np, pred_s_np, np.nan), cmap="YlOrRd", vmin=0, vmax=30)
    axes[2, 1].set_title("Pred Slope (°)")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"bev_ep{epoch}_b{batch_idx}.png"), dpi=120)
    plt.close(fig)


# ── Training ─────────────────────────────────────────────────────────

def train():
    print(f"Device: {DEVICE}")

    # Datasets
    train_ds = BEVDataset(DATA_ROOT, split="train", img_size=IMG_SIZE)
    val_ds   = BEVDataset(DATA_ROOT, split="val",   img_size=IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = BEVPredictionNet(num_classes=NUM_CLASSES, embed_dim=EMBED_DIM).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params / 1e6:.2f}M parameters")

    optimiser = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS)

    # Class weights (from previous analysis, clamped)
    class_weights = torch.tensor([
        0.28, 0.09, 0.05, 1.00, 50.0, 4.16, 21.3,
        0.28, 50.0, 19.1, 19.8, 0.53, 0.29,
    ], dtype=torch.float32, device=DEVICE)

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────
        model.train()
        train_losses = {"sem": 0, "height": 0, "slope": 0, "total": 0}
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{NUM_EPOCHS}")

        for batch_idx, batch in enumerate(pbar):
            imgs = batch["image"].to(DEVICE)
            K    = batch["intrinsics"].to(DEVICE)
            T    = batch["extrinsics"].to(DEVICE)

            gt_sem   = batch["bev_semantics"].to(DEVICE)
            gt_h     = batch["bev_height"].to(DEVICE)
            gt_s     = batch["bev_slope"].to(DEVICE)
            mask     = batch["bev_valid"].to(DEVICE)

            sem_logits, h_pred, s_pred, _ = model(imgs, K, T)

            l_sem = masked_semantic_loss(sem_logits, gt_sem, mask, class_weights)
            l_h   = masked_regression_loss(h_pred, gt_h, mask)
            l_s   = masked_regression_loss(s_pred, gt_s, mask)
            loss  = l_sem + ALPHA_HEIGHT * l_h + ALPHA_SLOPE * l_s

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            train_losses["sem"]    += l_sem.item()
            train_losses["height"] += l_h.item()
            train_losses["slope"]  += l_s.item()
            train_losses["total"]  += loss.item()

            pbar.set_postfix(
                sem=f"{l_sem.item():.3f}",
                h=f"{l_h.item():.3f}",
                s=f"{l_s.item():.3f}",
            )

            if batch_idx % VIS_INTERVAL == 0:
                visualise_bev(
                    sem_logits[0].detach(), gt_sem[0], h_pred[0].detach(), gt_h[0],
                    s_pred[0].detach(), gt_s[0], mask[0], batch["bev_fov_mask"][0],
                    epoch, batch_idx,
                )

        scheduler.step()
        n_train = max(1, len(train_loader))
        print(
            f"Epoch {epoch} Train | "
            f"Sem: {train_losses['sem']/n_train:.4f}  "
            f"Height: {train_losses['height']/n_train:.4f}  "
            f"Slope: {train_losses['slope']/n_train:.4f}  "
            f"Total: {train_losses['total']/n_train:.4f}"
        )

        # ── Validate ─────────────────────────────────────────────
        model.eval()
        val_losses = {"sem": 0, "height": 0, "slope": 0, "total": 0}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val {epoch}/{NUM_EPOCHS}"):
                imgs = batch["image"].to(DEVICE)
                K    = batch["intrinsics"].to(DEVICE)
                T    = batch["extrinsics"].to(DEVICE)

                gt_sem = batch["bev_semantics"].to(DEVICE)
                gt_h   = batch["bev_height"].to(DEVICE)
                gt_s   = batch["bev_slope"].to(DEVICE)
                mask   = batch["bev_valid"].to(DEVICE)

                sem_logits, h_pred, s_pred, _ = model(imgs, K, T)

                l_sem = masked_semantic_loss(sem_logits, gt_sem, mask, class_weights)
                l_h   = masked_regression_loss(h_pred, gt_h, mask)
                l_s   = masked_regression_loss(s_pred, gt_s, mask)
                loss  = l_sem + ALPHA_HEIGHT * l_h + ALPHA_SLOPE * l_s

                val_losses["sem"]    += l_sem.item()
                val_losses["height"] += l_h.item()
                val_losses["slope"]  += l_s.item()
                val_losses["total"]  += loss.item()

        n_val = max(1, len(val_loader))
        avg_val = val_losses["total"] / n_val
        print(
            f"Epoch {epoch} Val   | "
            f"Sem: {val_losses['sem']/n_val:.4f}  "
            f"Height: {val_losses['height']/n_val:.4f}  "
            f"Slope: {val_losses['slope']/n_val:.4f}  "
            f"Total: {avg_val:.4f}"
        )

        # ── Checkpoint ───────────────────────────────────────────
        os.makedirs(CKPT_DIR, exist_ok=True)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimiser.state_dict(),
                "val_loss": avg_val,
            }, os.path.join(CKPT_DIR, "best.pth"))
            print(f"  ★ Best model saved (val loss {avg_val:.4f})")

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimiser.state_dict(),
            }, os.path.join(CKPT_DIR, f"epoch_{epoch}.pth"))

    print("Training complete.")


if __name__ == "__main__":
    train()
