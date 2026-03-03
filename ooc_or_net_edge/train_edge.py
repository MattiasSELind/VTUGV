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
import matplotlib.pyplot as plt

from off_road_occ_net_edge import OffRoadOccNetEdge
from occ_dataset import OffRoadOccDataset

# --- Configuration ---
data_root = os.path.join(os.path.expanduser("~"), "Downloads", "Sample Dataset With Semantic Annotations")
batch_size = 16 # Increased batch size for better gradient stability and batch norm performance
learning_rate = 1e-4
num_epochs = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Params
num_semantic_classes = 13
dinov2_feat_dim = 384
embed_dim = 128
cameras = ["0000"]

def masked_semantic_loss(pred_logits, target_labels, valid_mask, class_weights=None):
    """
    Computes CrossEntropyLoss only on the valid BEV pixels where we have depth info.
    pred_logits: [B, num_classes, Y, X]
    target_labels: [B, Y, X]
    valid_mask: [B, Y, X] (Bool)
    class_weights: [num_classes] tensor for balanced cross entropy
    """
    loss = F.cross_entropy(pred_logits, target_labels, weight=class_weights, reduction='none', ignore_index=0)
    # Apply mask
    masked_loss = loss * valid_mask.float()
    
    # Average only over valid pixels. Avoid div by zero.
    num_valid = valid_mask.sum()
    if num_valid > 0:
        return masked_loss.sum() / num_valid
    else:
        return torch.tensor(0.0, device=pred_logits.device, requires_grad=True)

def masked_feature_loss(pred_feat, target_feat, valid_mask):
    """
    Computes Cosine Similarity and L1 loss for DINOv2 feature distillation
    only on the valid BEV pixels where we have depth info.
    """
    # Cosine Similarity: higher is better, so loss is 1 - cos_sim
    # F.cosine_similarity computes along dim 1 -> [B, Y, X]
    cos_sim = F.cosine_similarity(pred_feat, target_feat, dim=1)
    cos_loss = 1.0 - cos_sim
    
    # L1 Loss
    l1_loss = F.l1_loss(pred_feat, target_feat, reduction='none').mean(dim=1) # [B, Y, X]
    
    # Combined Distillation Loss
    loss = cos_loss + 0.5 * l1_loss
    
    # Apply mask
    masked_loss = loss * valid_mask.float()
    
    num_valid = valid_mask.sum()
    if num_valid > 0:
        return masked_loss.sum() / num_valid
    else:
        return torch.tensor(0.0, device=pred_feat.device, requires_grad=True)


def masked_occupancy_loss(pred_occ_logits, target_occ, mask):
    """
    Binary Cross Entropy loss for occupancy prediction, using logits for numerical stability.
    """
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_fn(pred_occ_logits.squeeze(1), target_occ)
    return (loss * mask).sum() / (mask.sum() + 1e-6)

def masked_cost_loss(pred_cost, target_cost, mask):
    """
    Smooth L1 loss for continuous traversability cost regression [0, 1].
    """
    loss = F.smooth_l1_loss(pred_cost.squeeze(1), target_cost, reduction='none')
    return (loss * mask).sum() / (mask.sum() + 1e-6)

def masked_depth_loss(pred_depths_2d, gt_lidar_depths, valid_mask):
    """
    Smooth L1 (Huber) Loss for explicit continuous depth regression.
    pred_depths_2d: [B, 1, fH, fW] Predicted Depth
    gt_lidar_depths: [B, 1, fH, fW] Lidar depth projected to feature scale
    valid_mask: [B, 1, fH, fW] Mask denoting where lidar hits exist
    """
    loss = F.smooth_l1_loss(pred_depths_2d, gt_lidar_depths, reduction='none')
    # Apply mask
    masked_loss = loss * valid_mask.float()
    
    num_valid = valid_mask.sum()
    if num_valid > 0:
        return masked_loss.sum() / num_valid
    else:
        return torch.tensor(0.0, device=pred_depths_2d.device, requires_grad=True)

def visualize_bev(pred_sem, gt_sem, pred_occ, gt_occ, pred_cost, gt_cost, epoch, batch_idx, save_dir="bev_visualizations"):
    """
    Saves a visualization comparing the predicted multi-task BEV outputs to Ground Truth.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get predictions
    pred_sem_classes = torch.argmax(pred_sem, dim=0).cpu().numpy()
    gt_sem_classes = gt_sem.cpu().numpy()
    
    pred_occ_binary = (torch.sigmoid(pred_occ.squeeze(0)) > 0.5).cpu().numpy()
    gt_occ_binary = gt_occ.cpu().numpy()
    
    pred_cost_map = pred_cost.squeeze(0).cpu().numpy()
    gt_cost_map = gt_cost.cpu().numpy()
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    
    # Row 0: Occupancy
    axes[0, 0].imshow(gt_occ_binary, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title("GT Occupancy")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_occ_binary, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title("Predicted Occupancy (>0.5)")
    axes[0, 1].axis('off')
    
    # Row 1: Semantics
    axes[1, 0].imshow(gt_sem_classes, cmap='tab20', vmin=0, vmax=13)
    axes[1, 0].set_title("GT Semantics")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_sem_classes, cmap='tab20', vmin=0, vmax=13)
    axes[1, 1].set_title("Predicted Semantics")
    axes[1, 1].axis('off')
    
    # Row 2: Traversability Cost
    axes[2, 0].imshow(gt_cost_map, cmap='magma', vmin=0.0, vmax=1.0)
    axes[2, 0].set_title("GT Cost Map [0-1]")
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(pred_cost_map, cmap='magma', vmin=0.0, vmax=1.0)
    axes[2, 1].set_title("Predicted Cost Map [0-1]")
    axes[2, 1].axis('off')
    
    plt.suptitle(f"Multi-Task Edge Inference | Epoch {epoch}, Batch {batch_idx}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"bev_ep{epoch}_b{batch_idx}.png"))
    plt.close(fig)

def train():
    print(f"--- Starting Depth-Guided BEV Training on {device} ---")
    
    # 1. Dataset & DataLoader
    seq_len = 2 # Short sequences for Recurrent Temporal Fusion
    if os.path.exists(data_root):
        train_dataset = OffRoadOccDataset(data_root, split="train", cameras=cameras, seq_len=seq_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        print(f"Loaded {len(train_dataset)} training samples.")
    else:
        print(f"ERROR: Data root {data_root} not found. Ensure dataset and depth maps exist.")
        return

    # 2. Model Initialization
    print("Initializing OffRoadOccNetEdge Model...")
    occ_net = OffRoadOccNetEdge(
        num_semantic_classes=num_semantic_classes, 
        dinov2_feat_dim=dinov2_feat_dim, 
        embed_dim=embed_dim
    ).to(device)
    
    optimizer = AdamW(occ_net.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Load actual Class Weights calculated from class_statistics.txt
    print("Setting up Class Weights for Balanced Cross Entropy...")
    
    # Inverted frequency weights clamped to max 50.0
    actual_weights = [
        0.2818,   # Class 0 dirt road  -> 🟤 brown
        0.0890,   # Class 1 grass      -> 🟢 lawn green
        0.0513,   # Class 2 tree       -> 🌲 forest green
        1.0000,   # Class 3 bush       -> 🟢 green
        50.0000,  # Class 4 rock       -> ⬜ gray
        4.1579,   # Class 5 mud        -> 🟤 dark brown
        21.3335,  # Class 6 water      -> 🔵 blue
        0.2785,   # Class 7 sky        -> 🔵 light blue
        50.0000,  # Class 8 vehicle    -> 🔵 dark blue
        19.0855,  # Class 9 person     -> 🔴 red
        19.7631,  # Class 10 building   -> ⬛ dark gray
        0.5326,   # Class 11 fence      -> ⬜ blush
        0.2916,   # Class 12 terrain    -> 🟡 tan
    ]
    
    class_weights = torch.tensor(actual_weights, dtype=torch.float32, device=device)

    # 3. Training Loop
    occ_net.train()
    
    for epoch in range(num_epochs):
        total_sem_loss = 0.0
        total_feat_loss = 0.0
        total_occ_loss = 0.0
        total_cost_loss = 0.0
        
        # tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            
            # Skip batches without depth-guided BEV ground truth
            if "gt_bev_semantic" not in batch:
                continue
                
            optimizer.zero_grad()
            
            # Extract sequence properties
            images_seq = batch["images"].to(device)           # [S, B, 3, H, W]
            poses_seq = batch["poses"].to(device)             # [S, B, 4, 4] T_veh_camera
            intrinsics_seq = batch["intrinsics"].to(device)   # [S, B, 4, 4]
            world_poses_seq = batch["world_pose"].to(device)  # [S, B, 4, 4] T_world_veh
            
            # Ground truth targets are only evaluated at the FINAL step S-1
            gt_bev_sem = batch["gt_bev_semantic"].to(device) # [B, 50, 50]
            gt_bev_valid = batch["gt_bev_valid"].to(device)  # [B, 50, 50]
            gt_bev_feat = batch["gt_bev_feat"].to(device)    # [B, 384, 50, 50]
            gt_bev_occ = batch["gt_bev_occ"].to(device)      # [B, 50, 50]
            gt_bev_cost = batch["gt_bev_cost"].to(device)    # [B, 50, 50]
            
            # 2D Depth targets from LiDAR (only needed for the final frame S-1 during distillation)
            gt_lidar_depths = batch["gt_lidar_depth_2d"].to(device) # [B, 1, fH, fW]
            gt_lidar_valid = batch["gt_lidar_valid"].to(device)     # [B, 1, fH, fW]
            
            S = images_seq.shape[0]
            prev_state = None
            prev_world_pose = None
            
            # --- 1. Forward Pass (Unroll Sequence) ---
            for t in range(S):
                # Extract instantaneous inputs
                imgs = images_seq[t]
                poses = poses_seq[t]
                intrinsics = intrinsics_seq[t]
                curr_world_pose = world_poses_seq[t]
                
                # Pass through the Temporally-Fused model
                semantic_bev, feature_bev, pred_depths_2d, curr_state = occ_net(
                    imgs, intrinsics, poses, 
                    prev_bev=prev_state, 
                    prev_pose=prev_world_pose, 
                    curr_pose=curr_world_pose
                )
                
                # Update recurrent state variables
                prev_state = curr_state
                prev_world_pose = curr_world_pose
                
            # --- 2. Compute Masked Loss (Only on Final Timestep t = S - 1) ---
            # Occupancy Loss
            o_loss = masked_occupancy_loss(occ_logits, gt_bev_occ, gt_bev_valid)
            
            # Semantic Loss
            s_loss = masked_semantic_loss(semantic_bev, gt_bev_sem, gt_bev_valid, class_weights=class_weights)
            
            # Feature matching loss
            f_loss = masked_feature_loss(feature_bev, gt_bev_feat, gt_bev_valid)
            
            # Traversability Cost loss
            c_loss = masked_cost_loss(pred_cost_map, gt_bev_cost, gt_bev_valid)
            
            # Explicit Depth Loss from LiDAR projected into the 2D feature grid
            # pred_depths_2d is [B, 1, fH, fW] output by the Auxiliary Depth Head
            d_loss = masked_depth_loss(pred_depths_2d, gt_lidar_depths, gt_lidar_valid)
            
            alpha_distill = 10.0 # DINOv2 cosine distillation usually needs a higher weight to balance CE
            alpha_occ = 5.0      # Balance BCE scale
            alpha_cost = 5.0     # Balance Cost scale
            alpha_depth = 3.0    # LiDAR Explicit Depth Regression
            
            total_loss = s_loss + (alpha_distill * f_loss) + (alpha_occ * o_loss) + (alpha_cost * c_loss) + (alpha_depth * d_loss)
            
            # --- 3. Backpropagate ---
            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(occ_net.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Logging
            total_sem_loss += s_loss.item()
            total_feat_loss += f_loss.item()
            total_occ_loss += o_loss.item()
            total_cost_loss += c_loss.item()
            
            pbar.set_postfix({
                'Occ': f"{o_loss.item():.3f}", 
                'Sem': f"{s_loss.item():.3f}", 
                'Feat': f"{f_loss.item():.3f}", 
                'Cost': f"{c_loss.item():.3f}",
                'Depth': f"{d_loss.item():.3f}"
            })
            
            # Save a visualization every 50 batches
            if batch_idx % 50 == 0:
                # pass the first sample in the batch
                visualize_bev(
                    pred_sem=semantic_bev[0].detach(), gt_sem=gt_bev_sem[0],
                    pred_occ=occ_logits[0].detach(), gt_occ=gt_bev_occ[0],
                    pred_cost=pred_cost_map[0].detach(), gt_cost=gt_bev_cost[0],
                    epoch=epoch+1, batch_idx=batch_idx
                )

        # Epoch Summary
        avg_occ = total_occ_loss / max(1, len(train_loader))
        avg_sem = total_sem_loss / max(1, len(train_loader))
        avg_feat = total_feat_loss / max(1, len(train_loader))
        avg_cost = total_cost_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Avg Occ Loss: {avg_occ:.4f}, Avg Sem Loss: {avg_sem:.4f}, Avg Feat Loss: {avg_feat:.4f}, Avg Cost Loss: {avg_cost:.4f}")
        
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
