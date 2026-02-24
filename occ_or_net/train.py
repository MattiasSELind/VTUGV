"""
train.py
Self-Supervised Training Loop for OffRoadOccNet (QueryOcc architecture).
Utilizes QueryOccDataset to fetch synchronized data, constructs a 3D semantic/feature volume,
and supervises it using Novel View Synthesis (Volume Rendering) against DINOv2 features and SeMasks.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from occ_dataset import OffRoadOccDataset
from off_road_occ_net import OffRoadOccNet
from volume_renderer import VolumeRenderer

# --- Configuration ---
data_root = os.path.join(os.path.expanduser("~"), "Downloads", "VTUGV_pointclouds_outdoors")
batch_size = 2
learning_rate = 1e-4
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Params
num_semantic_classes = 150
dinov2_feat_dim = 384
embed_dim = 256
cameras = ["front_left", "front_right", "rear_center"]

def feature_metric_loss(rendered_feats, ground_truth_feats):
    """
    Cosine similarity loss between rendered features and DINOv2 ground truth.
    Both should be shape [B, C, H, W].
    """
    # Normalize features along channel dimension
    render_norm = F.normalize(rendered_feats, p=2, dim=1)
    gt_norm = F.normalize(ground_truth_feats, p=2, dim=1)
    
    # Cosine similarity (1 is exact match, -1 is opposite)
    cos_sim = torch.sum(render_norm * gt_norm, dim=1) # [B, H, W]
    
    # We want to maximize similarity, so loss = 1 - sim
    loss = 1.0 - cos_sim.mean()
    return loss

def semantic_loss(rendered_logits, ground_truth_labels):
    """
    Cross Entropy Loss for semantic rendering.
    rendered_logits: [B, num_classes, H, W]
    ground_truth_labels: [B, H, W] (LongTensor of class indices)
    """
    # Cross entropy inherently applies log-softmax
    loss = F.cross_entropy(rendered_logits, ground_truth_labels, ignore_index=0) # Assuming 0 is background/unknown
    return loss

def train():
    print(f"--- Starting Self-Supervised QueryOcc Training on {device} ---")
    
    # 1. Dataset & DataLoader (Only if data exists, otherwise mock it for testing)
    if os.path.exists(data_root):
        train_dataset = OffRoadOccDataset(data_root, split="train", cameras=cameras)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        print(f"Loaded {len(train_dataset)} training samples.")
    else:
        print(f"WARNING: Data root {data_root} not found. Running with mock dataloader.")
        # Create a tiny mock dataloader for dry runs
        train_loader = [ {
            "images": torch.randn(batch_size, 3, 3, 256, 512),
            "features": torch.randn(batch_size, 3, dinov2_feat_dim, 18, 36),
            "semantics": torch.randint(0, num_semantic_classes, (batch_size, 3, 256, 512)),
            "poses": torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, 3, 1, 1),
            "intrinsics": torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, 3, 1, 1),
            "basename": ["mock"] * batch_size
        } for _ in range(5) ]

    # 2. Models
    print("Initializing Models...")
    occ_net = OffRoadOccNet(
        num_semantic_classes=num_semantic_classes, 
        dinov2_feat_dim=dinov2_feat_dim, 
        embed_dim=embed_dim
    ).to(device)
    
    renderer = VolumeRenderer(
        x_bounds=[-50, 50], y_bounds=[-50, 50], z_bounds=[-2, 6],
        resolution=1.0, num_samples=64
    ).to(device)
    
    optimizer = AdamW(occ_net.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 3. Training Loop
    # In self-supervised volume rendering, we often train by:
    #   - Building the 3D grid from N-1 cameras (or time t-1)
    #   - Rendering the novel view of the Nth camera (or time t)
    # For simplicity here: we build the grid using ALL cameras, but render views 
    # to reconstruct the target cameras.
    
    occ_net.train()
    renderer.train()
    
    for epoch in range(num_epochs):
        total_feat_loss = 0.0
        total_sem_loss = 0.0
        
        # tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            
            # Move to device
            imgs = batch["images"].to(device)         # [B, num_cams, 3, H, W]
            feats = batch["features"].to(device)      # [B, num_cams, D, Hp, Wp]
            sems = batch["semantics"].to(device)      # [B, num_cams, H, W]
            poses = batch["poses"].to(device)         # [B, num_cams, 4, 4] T_world_camera
            intrinsics = batch["intrinsics"].to(device) # [B, num_cams, 4, 4]
            
            optimizer.zero_grad()
            
            # --- 1. Construct 3D Volume ---
            # We pass all images through the backbone and lift them into a unified 3D grid.
            semantic_vol, feature_vol = occ_net(imgs, intrinsics, poses)
            # semantic_vol: [B, num_classes, Z, Y, X]
            # feature_vol: [B, dinov2_feat_dim, Z, Y, X]
            
            batch_feat_loss = 0.0
            batch_sem_loss = 0.0
            
            # --- 2. Render Novel Views & Compute Loss ---
            # We iterate over each camera, rendering the novel view from its pose
            # and comparing it against the true features/semantics collected by that camera.
            num_cams = imgs.shape[1]
            for target_cam_idx in range(num_cams):
                K_target = intrinsics[:, target_cam_idx]
                T_world_target = poses[:, target_cam_idx]
                
                # Render using Alpha Compositing
                # Target feature map resolution depends on DINOv2 patch grid
                H_feat, W_feat = feats.shape[-2], feats.shape[-1]
                H_sem, W_sem = sems.shape[-2], sems.shape[-1]
                
                # We need to render the semantic map at full resolution, and features at patch resolution
                # (Or render both at full resolution and interpolate features up). 
                # Let's render both at the DINOv2 feature resolution for speed during training.
                
                rendered_sem, rendered_feat, _ = renderer.render(
                    H_feat, W_feat, K_target, T_world_target, 
                    semantic_vol, feature_vol
                )
                # rendered_sem: [B, num_classes, H_feat, W_feat]
                # rendered_feat: [B, feat_dim, H_feat, W_feat]
                
                # Ground truth for this camera
                gt_feat = feats[:, target_cam_idx] # [B, feat_dim, H_feat, W_feat]
                
                # Downsample semantic GT to match feature grid resolution
                gt_sem = sems[:, target_cam_idx].float().unsqueeze(1) # [B, 1, H_sem, W_sem]
                gt_sem_down = F.interpolate(gt_sem, size=(H_feat, W_feat), mode='nearest').squeeze(1).long() # [B, H_feat, W_feat]
                
                # Losses
                f_loss = feature_metric_loss(rendered_feat, gt_feat)
                s_loss = semantic_loss(rendered_sem, gt_sem_down)
                
                batch_feat_loss += f_loss
                batch_sem_loss += s_loss
                
            # Average over all targeting cameras
            batch_feat_loss = batch_feat_loss / num_cams
            batch_sem_loss = batch_sem_loss / num_cams
            
            # Total Loss (can weight these later)
            w_feat, w_sem = 1.0, 1.0
            total_loss = w_feat * batch_feat_loss + w_sem * batch_sem_loss
            
            # --- 3. Backpropagate ---
            total_loss.backward()
            
            # Optional: gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(occ_net.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Logging
            total_feat_loss += batch_feat_loss.item()
            total_sem_loss += batch_sem_loss.item()
            
            pbar.set_postfix({
                'Feat_L': f"{batch_feat_loss.item():.4f}", 
                'Sem_L': f"{batch_sem_loss.item():.4f}"
            })

        # Epoch Summary
        avg_feat = total_feat_loss / len(train_loader)
        avg_sem = total_sem_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Avg Feat Loss: {avg_feat:.4f}, Avg Sem Loss: {avg_sem:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/offroad_occnet_ep{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': occ_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    print("Training finished.")

if __name__ == "__main__":
    train()
