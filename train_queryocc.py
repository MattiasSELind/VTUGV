"""
=============================================================================
QueryOcc Training Script
=============================================================================
Trains the QueryOccNet using self-supervised volume rendering.
"""

import os
import csv
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Import new components
from queryocc_components import QueryOccNet

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED_DIR = BASE_DIR / "extracted"
IMAGE_DIR = EXTRACTED_DIR / "images" / "multisense_left_image_rect_color"
DEPTH_DIR = EXTRACTED_DIR / "depth"
ODOM_DIR  = EXTRACTED_DIR / "odometry"
CALIB_DIR = EXTRACTED_DIR / "calibration"
DEBUG_DIR = EXTRACTED_DIR / "queryocc_debug"

DEBUG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE  = 2e-4
NUM_EPOCHS     = 30
BATCH_SIZE     = 2


# ── Data Parsing (Reused from pipeline_4d.py) ──────────────────────────────
def parse_intrinsics(filepath):
    # Default values
    width, height = 1024, 544
    K = np.eye(3)
    D = np.zeros(5)
    
    if not os.path.exists(filepath):
        print(f"Warning: Intrinsics file not found: {filepath}")
        return K, None, D, (width, height)

    with open(filepath, "r") as f:
        content = f.read()
    
    # Try to find the specific section, else use the whole content or first section
    sections = content.split("---")
    target_section = sections[0]
    for section in sections:
        if "left_image_rect_color" in section:
            target_section = section
            break
            
    for line in target_section.split("\n"):
        line = line.strip()
        if line.startswith("K:"):
            try:
                k_vals = [float(x) for x in line[2:].strip().strip("[]").split(",")]
                K = np.array(k_vals).reshape(3, 3)
            except: pass
        elif line.startswith("D:"):
            try:
                d_vals = [float(x) for x in line[2:].strip().strip("[]").split(",")]
                D = np.array(d_vals)
            except: pass
        elif line.startswith("width:"): 
            try: 
                val = int(line.split(":")[1].strip())
                if val > 0: width = val
            except: pass
        elif line.startswith("height:"): 
            try: 
                val = int(line.split(":")[1].strip())
                if val > 0: height = val
            except: pass
            
    return K, None, D, (width, height)

def parse_extrinsics(filepath):
    transforms = {}
    if not os.path.exists(filepath):
        return {"vehicle_to_left_optical": np.eye(4)}
        
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
        
    T_v2head, T_head2optical = None, None
    for entry in data["transform_params"]:
        t, q = np.array(entry["translation"]), entry["quaternion"]
        R = Rotation.from_quat(q).as_matrix()
        T = np.eye(4); T[:3,:3]=R; T[:3,3]=t
        if entry["from_frame"]=="vehicle" and "head" in entry["to_frame"]: T_v2head=T
        if "head" in entry["from_frame"] and "left_camera_optical" in entry["to_frame"]: T_head2optical=T
    
    if T_v2head is not None and T_head2optical is not None:
        transforms["vehicle_to_left_optical"] = T_head2optical @ T_v2head
    else:
        transforms["vehicle_to_left_optical"] = np.eye(4)
    return transforms

def parse_odometry(filepath):
    timestamps, poses = [], []
    with open(filepath, "r") as f:
        for row in csv.DictReader(f):
            timestamps.append(float(row["timestamp_sec"]) + float(row["timestamp_nsec"])*1e-9)
            pos = np.array([float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])])
            quat = np.array([float(row["orient_x"]), float(row["orient_y"]), float(row["orient_z"]), float(row["orient_w"])])
            T = np.eye(4); T[:3,:3]=Rotation.from_quat(quat).as_matrix(); T[:3,3]=pos
            poses.append(T)
    return np.array(timestamps), poses

def parse_image_timestamps(filepath):
    frame_indices, timestamps, filenames = [], [], []
    with open(filepath, "r") as f:
        for row in csv.DictReader(f):
            frame_indices.append(int(row["frame_index"]))
            timestamps.append(float(row["timestamp_sec"]) + float(row["timestamp_nsec"])*1e-9)
            filenames.append(row["filename"])
    return frame_indices, np.array(timestamps), filenames

def match_poses_to_frames(img_ts, odom_ts, odom_poses):
    matched = []
    for ts in img_ts:
        idx = np.argmin(np.abs(odom_ts - ts))
        matched.append(odom_poses[idx])
    return matched, None


# ── Dataset ──────────────────────────────────────────────────────────────────
class TartanDrive4DDataset(Dataset):
    def __init__(self, image_dir, frame_indices, filenames, matched_poses, 
                 img_timestamps, K, img_height, img_width):
        self.image_dir = Path(image_dir)
        self.img_height, self.img_width = img_height, img_width
        self.K = torch.from_numpy(K.astype(np.float32))
        self.triplets = []
        
        for i in range(1, len(frame_indices)-1):
            dt_prev = img_timestamps[i] - img_timestamps[i-1]
            dt_next = img_timestamps[i+1] - img_timestamps[i]
            if dt_prev > 0.5 or dt_next > 0.5: continue
            
            self.triplets.append({
                "prev": (str(self.image_dir/filenames[i-1]), matched_poses[i-1]),
                "curr": (str(self.image_dir/filenames[i]),   matched_poses[i]),
                "next": (str(self.image_dir/filenames[i+1]), matched_poses[i+1]),
            })
            
    def __len__(self): return len(self.triplets)
    
    def __getitem__(self, idx):
        s = self.triplets[idx]
        def load_img(path):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            return torch.from_numpy(img).permute(2,0,1)
            
        imgs = [load_img(s[k][0]) for k in ["prev", "curr", "next"]]
        poses = [torch.from_numpy(s[k][1].astype(np.float32)) for k in ["prev", "curr", "next"]]
        
        return {"images": imgs, "poses": poses, "K": self.K}

# ── Loss Functions ───────────────────────────────────────────────────────────
def ssim(x, y, window_size=3):
    C1 = 0.01**2; C2 = 0.03**2
    pad = window_size//2
    mu_x = F.avg_pool2d(x, window_size, 1, pad)
    mu_y = F.avg_pool2d(y, window_size, 1, pad)
    sigma_x = F.avg_pool2d(x**2, window_size, 1, pad) - mu_x**2
    sigma_y = F.avg_pool2d(y**2, window_size, 1, pad) - mu_y**2
    sigma_xy = F.avg_pool2d(x*y, window_size, 1, pad) - mu_x*mu_y
    ssim_n = (2*mu_x*mu_y + C1)*(2*sigma_xy + C2)
    ssim_d = (mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2)
    return (ssim_n / ssim_d).mean(1, keepdim=True)

def reconstruction_loss(pred_rgb, target_rgb, alpha=0.85):
    l1 = (pred_rgb - target_rgb).abs().mean(1, keepdim=True)
    ssim_val = ssim(pred_rgb, target_rgb)
    loss = alpha * (1 - ssim_val)/2 + (1 - alpha) * l1
    return loss.mean()

# ── Main Training Loop ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Preparing QueryOcc Training ──")
    
    # 1. Load Data/Calibration
    K, _, _, img_size = parse_intrinsics(CALIB_DIR / "multisense_intrinsics.txt")
    print(f"  Image Size: {img_size}")
    
    extrinsics = parse_extrinsics(CALIB_DIR / "extrinsics.yaml")
    T_vh_cam = extrinsics["vehicle_to_left_optical"]
    T_cam_to_veh = np.linalg.inv(T_vh_cam)
    T_cam_to_veh_torch = torch.from_numpy(T_cam_to_veh.astype(np.float32)).to(DEVICE)
    
    odom_ts, odom_poses = parse_odometry(ODOM_DIR / "tartanvo_odom.csv")
    frame_indices, img_ts, img_names = parse_image_timestamps(IMAGE_DIR / "timestamps.csv")
    matched_poses, _ = match_poses_to_frames(img_ts, odom_ts, odom_poses)
    
    # 2. Dataset
    dataset = TartanDrive4DDataset(
        IMAGE_DIR, frame_indices, img_names, matched_poses, img_ts, K, img_size[1], img_size[0]
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"  Dataset: {len(dataset)} triplets")
    
    # Train at lower resolution for speed
    TRAIN_H, TRAIN_W = 136, 256 # 4x Downsample
    model = QueryOccNet(img_size=(TRAIN_H, TRAIN_W), num_bins=32).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n── Starting Training (Low Res for Demo) ──")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for i, batch in enumerate(dataloader):
            # Move to device and downsample
            images = [F.interpolate(img.to(DEVICE), size=(TRAIN_H, TRAIN_W), mode='area') for img in batch["images"]]
            poses  = [p.to(DEVICE) for p in batch["poses"]]
            
            # Scale Intrinsics (approx)
            K_batch = batch["K"].to(DEVICE).clone()
            K_batch[:, 0, :] *= (TRAIN_W / img_size[0])
            K_batch[:, 1, :] *= (TRAIN_H / img_size[1])
            
            optimizer.zero_grad()
            
            # Forward: Render Current Frame
            pred_depth, pred_rgb = model(images, poses, K_batch, T_cam_to_veh_torch)
            
            # Loss: Compare rendered with downsampled target
            img_curr = images[1]
            loss = reconstruction_loss(pred_rgb, img_curr)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), "queryocc_checkpoint.pth")
        
        # Debug Visualization
        if (epoch+1) % 1 == 0:
            with torch.no_grad():
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(images[1][0].cpu().permute(1,2,0).numpy())
                ax[0].set_title("Target RGB")
                ax[1].imshow(pred_rgb[0].cpu().permute(1,2,0).numpy().clip(0,1))
                ax[1].set_title("Rendered RGB")
                ax[2].imshow(pred_depth[0,0].cpu().numpy(), cmap='plasma')
                ax[2].set_title("Rendered Depth")
                plt.savefig(DEBUG_DIR / f"epoch_{epoch+1}.png")
                plt.close()
