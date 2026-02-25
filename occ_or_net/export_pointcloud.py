"""
export_pointcloud.py
Loads a trained OffRoadOccNet checkpoint, runs a forward pass on a dataset sample,
and exports the predicted 3D Semantic Occupancy Volume to a .ply Point Cloud file
for visualization in tools like MeshLab or CloudCompare.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from occ_dataset import OffRoadOccDataset
from off_road_occ_net import OffRoadOccNet

# --- Configuration ---
CHECKPOINT_PATH = "checkpoints/offroad_occnet_ep10.pth"  # E.g., "checkpoints/offroad_occnet_ep5.pth"
DATA_ROOT = os.path.join(os.path.expanduser("~"), "Downloads", "Data_Outdoors")
OUTPUT_FILE = "semantic_occupancy.ply"

# Match your training config
NUM_CLASSES = 14
FEAT_DIM = 384
EMBED_DIM = 256
RESOLUTION = 0.5  # Voxel size in meters
X_BOUNDS = [-50, 50]
Y_BOUNDS = [-50, 50]
Z_BOUNDS = [-2, 6]

# A random color map for visualizing semantic classes (150 classes)
np.random.seed(42)
COLOR_MAP = np.random.rand(NUM_CLASSES, 3)
# Let's make class 0 (background/unknown) black or transparent
COLOR_MAP[0] = [0.0, 0.0, 0.0]

def extract_point_cloud(semantic_volume, density_threshold=0.5):
    """
    Converts the voxel volume [num_classes, Z, Y, X] into a Point Cloud.
    We classify a voxel as 'occupied' if its maximum semantic logit is above a threshold.
    """
    # semantic_volume shape: [C, Z, Y, X]
    device = semantic_volume.device
    C, Z, Y, X = semantic_volume.shape
    
    # 1. Get the max class and its confidence/density
    max_logits, max_classes = torch.max(semantic_volume, dim=0) # [Z, Y, X]
    
    # 2. Filter out low confidence voxels (empty space) and background class
    occupied_mask = (max_logits > density_threshold) & (max_classes > 0)
    
    # Get physical coordinates for the grid
    x_coords = torch.linspace(X_BOUNDS[0], X_BOUNDS[1], X, device=device)
    y_coords = torch.linspace(Y_BOUNDS[0], Y_BOUNDS[1], Y, device=device)
    z_coords = torch.linspace(Z_BOUNDS[0], Z_BOUNDS[1], Z, device=device)
    
    grid_z, grid_y, grid_x = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    
    # Mask to keep only the points that exist
    points_x = grid_x[occupied_mask].cpu().numpy()
    points_y = grid_y[occupied_mask].cpu().numpy()
    points_z = grid_z[occupied_mask].cpu().numpy()
    
    points = np.stack([points_x, points_y, points_z], axis=-1)
    
    # Get colors for the surviving points
    classes = max_classes[occupied_mask].cpu().numpy()
    colors = COLOR_MAP[classes]
    
    return points, colors

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    # 1. Initialize Network
    model = OffRoadOccNet(
        num_semantic_classes=NUM_CLASSES, 
        dinov2_feat_dim=FEAT_DIM, 
        embed_dim=EMBED_DIM
    ).to(device)
    
    # Load Checkpoint weights if provided
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("WARNING: No valid checkpoint found. Using randomly initialized weights.")
        print("The output point cloud will be noise until you load a trained .pth file.")
        
    model.eval()
    
    # 2. Load one sample from dataset
    dataset = OffRoadOccDataset(DATA_ROOT, split="val", cameras=["front_left", "rear_center"])
    if len(dataset) == 0:
        print("Dataset empty or not found!")
        return
        
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    
    imgs = batch["images"].to(device)
    intrinsics = batch["intrinsics"].to(device)
    poses = batch["poses"].to(device)
    
    print(f"Running forward pass for frame: {batch['basename'][0]}")
    
    with torch.no_grad():
        # semantic_vol: [B, C, Z, Y, X]
        semantic_vol, _ = model(imgs, intrinsics, poses)
        
    # We only care about batch index 0
    semantic_vol_single = semantic_vol[0] 
    
    # 3. Extract Point Cloud
    print("Extracting occupied voxels into Point Cloud...")
    # You might need to tune this threshold depending on your loss curve and data!
    points, colors = extract_point_cloud(semantic_vol_single, density_threshold=0.5) 
    
    if len(points) == 0:
        print("No voxels passed the density threshold. Try lowering it or training more.")
        return
        
    print(f"Generated {len(points)} occupied points.")
    
    # 4. Export Manually to PLY
    with open(OUTPUT_FILE, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for p, c in zip(points, colors):
            r, g, b = (c * 255).astype(int)
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {r} {g} {b}\n")
    
    print(f"Point cloud saved to {OUTPUT_FILE}!")
    
if __name__ == "__main__":
    main()
