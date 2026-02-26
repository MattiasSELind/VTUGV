"""
train_nano.py
Self-Supervised / Depth-Supervised Training Loop for OffRoadOccNetNano.
Utilizes depth-guided pseudo-ground truth BEV maps for direct 2D supervision,
bypassing the need for computationally heavy 3D Ray Casting.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from occ_dataset import OffRoadOccDataset
from off_road_occ_net_nano import OffRoadOccNetNano

# --- Configuration ---
data_root = os.path.join(os.path.expanduser("~"), "Downloads", "Data_Outdoors")
batch_size = 2
learning_rate = 1e-4
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Params
num_semantic_classes = 14
dinov2_feat_dim = 384
embed_dim = 128
cameras = ["front_left", "rear_center"]

def masked_semantic_loss(pred_logits, target_labels, valid_mask):
    """
    Computes CrossEntropyLoss only on the valid BEV pixels where we have depth info.
    pred_logits: [B, num_classes, Y, X]
    target_labels: [B, Y, X]
    valid_mask: [B, Y, X] (Bool)
    """
    loss = F.cross_entropy(pred_logits, target_labels, reduction='none', ignore_index=0)
    # Apply mask
    masked_loss = loss * valid_mask.float()
    
    # Average only over valid pixels. Avoid div by zero.
    num_valid = valid_mask.sum()
    if num_valid > 0:
        return masked_loss.sum() / num_valid
    else:
        return torch.tensor(0.0, device=pred_logits.device, requires_grad=True)

def train():
    print(f"--- Starting Depth-Guided BEV Training on {device} ---")
    
    # 1. Dataset & DataLoader
    if os.path.exists(data_root):
        train_dataset = OffRoadOccDataset(data_root, split="train", cameras=cameras)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        print(f"Loaded {len(train_dataset)} training samples.")
    else:
        print(f"ERROR: Data root {data_root} not found. Ensure dataset and depth maps exist.")
        return

    # 2. Model Initialization
    print("Initializing OffRoadOccNetNano Model...")
    occ_net = OffRoadOccNetNano(
        num_semantic_classes=num_semantic_classes, 
        dinov2_feat_dim=dinov2_feat_dim, 
        embed_dim=embed_dim
    ).to(device)
    
    optimizer = AdamW(occ_net.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 3. Training Loop
    occ_net.train()
    
    for epoch in range(num_epochs):
        total_sem_loss = 0.0
        
        # tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            
            # Skip batches without depth-guided BEV ground truth
            if "gt_bev_semantic" not in batch:
                continue
                
            # Move inputs to device
            imgs = batch["images"].to(device)           # [B, num_cams, 3, H, W]
            poses = batch["poses"].to(device)           # [B, num_cams, 4, 4] T_veh_camera
            intrinsics = batch["intrinsics"].to(device) # [B, num_cams, 4, 4]
            
            # Move Targets to device
            gt_bev_sem = batch["gt_bev_semantic"].to(device) # [B, 50, 50]
            gt_bev_valid = batch["gt_bev_valid"].to(device)  # [B, 50, 50]
            
            optimizer.zero_grad()
            
            # --- 1. Forward Pass (2D to BEV) ---
            # We pass all images through the backbone and map them into the BEV Grid
            semantic_bev, feature_bev = occ_net(imgs, intrinsics, poses)
            # semantic_bev: [B, num_classes, 50, 50]
            
            # --- 2. Compute Masked Loss ---
            # Since we collapsed the Z dimension, we directly compare the BEV output
            # against the depth-projected semantic BEV ground truth.
            s_loss = masked_semantic_loss(semantic_bev, gt_bev_sem, gt_bev_valid)
            
            # Feature matching loss (Optional, can be added if GT BEV Features are constructed)
            # f_loss = masked_feature_loss(feature_bev, gt_bev_feat, gt_bev_valid)
            
            total_loss = s_loss
            
            # --- 3. Backpropagate ---
            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(occ_net.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Logging
            total_sem_loss += s_loss.item()
            pbar.set_postfix({'Sem_Loss': f"{s_loss.item():.4f}"})

        # Epoch Summary
        avg_sem = total_sem_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Avg Sem Loss: {avg_sem:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/offroad_occnet_nano_ep{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': occ_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    print("Training finished.")

if __name__ == "__main__":
    train()
