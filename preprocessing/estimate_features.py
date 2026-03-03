"""
estimate_features.py
Extract dense visual features from images using DINOv2 (Meta).
Saves patch-level feature maps as .npz files for downstream 3D lifting.

DINOv2 produces rich per-patch features (384-dim for small, 768 for base)
that capture semantic, geometric, and textural information — ideal for
feature-based 3D point clouds and neural scene representations.
"""

import os
import glob
import numpy as np
from PIL import Image

# Set HF cache before importing torch
os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), "Downloads", "hf_cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Configuration
HOME = os.path.expanduser("~")
DATASET_DIR = os.path.join(HOME, "Downloads", "Sample Dataset With Semantic Annotations", "pylon_camera_node")
IMAGE_DIR = DATASET_DIR
OUTPUT_DIR = os.path.join(HOME, "Downloads", "Sample Dataset With Semantic Annotations", "pylon_camera_node", "features")

CAMERAS = ["0000"]

# Limit for testing — set to None for full dataset
LIMIT = None

# Optimization Configuration
BATCH_SIZE = 32
NUM_WORKERS = 8
VISUALIZE_FIRST_ONLY = True

# DINOv2 model size: "small" (384-dim, fast) or "base" (768-dim, richer)
MODEL_SIZE = "small"

# Input resolution for DINOv2 (must be multiple of 14 for ViT patch size)
# Original images are large (1920x1200). 
# We downscale to save training VRAM while keeping the aspect ratio roughly 1.6:1
# 504 / 14 = 36 width patches. 308 / 14 = 22 height patches.
INPUT_WIDTH = 504
INPUT_HEIGHT = 308

class ImageDataset(Dataset):
    def __init__(self, img_paths, target_height, target_width):
        self.img_paths = img_paths
        self.transform = T.Compose([
            T.Resize((target_height, target_width), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = Image.open(path).convert("RGB")
        orig_w, orig_h = image.size
        tensor = self.transform(image)
        return path, tensor, orig_w, orig_h

def load_dinov2(model_size="small", device="cpu"):
    """Load DINOv2 model from torch hub."""
    model_name = f"dinov2_vit{model_size[0]}14"
    print(f"Loading DINOv2: {model_name}")
    
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.to(device)
    model.eval()
    
    feat_dim = {"small": 384, "base": 768, "large": 1024, "giant": 1536}[model_size]
    return model, feat_dim

def create_pca_visualization(features, n_components=3):
    """
    Create a false-color RGB visualization of features using PCA.
    """
    H, W, D = features.shape
    flat = features.reshape(-1, D)
    
    mean = flat.mean(axis=0)
    centered = flat - mean
    
    if D < flat.shape[0]:
        cov = centered.T @ centered / flat.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        top3 = eigenvectors[:, -n_components:][:, ::-1]
    else:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        top3 = Vt[:n_components].T
    
    projected = centered @ top3
    
    for c in range(3):
        col = projected[:, c]
        mn, mx = col.min(), col.max()
        if mx > mn:
            projected[:, c] = (col - mn) / (mx - mn) * 255
        else:
            projected[:, c] = 128
            
    vis = projected.reshape(H, W, 3).astype(np.uint8)
    return vis

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, feat_dim = load_dinov2(MODEL_SIZE, device)
    
    # Feature map dimensions (patches)
    patch_h = INPUT_HEIGHT // 14
    patch_w = INPUT_WIDTH // 14
    
    for cam_name in CAMERAS:
        cam_in_dir = os.path.join(IMAGE_DIR, cam_name)
        cam_out_dir = os.path.join(OUTPUT_DIR, cam_name)
        
        if not os.path.isdir(cam_in_dir):
            print(f"\nCamera folder not found: {cam_in_dir}")
            continue
            
        os.makedirs(cam_out_dir, exist_ok=True)
        
        img_paths = sorted(
            glob.glob(os.path.join(cam_in_dir, "*.jpg")) +
            glob.glob(os.path.join(cam_in_dir, "*.png"))
        )
        
        # Filter out already processed images to avoid dataset loading them
        unprocessed_paths = []
        for p in img_paths:
            basename = os.path.basename(p)
            name_no_ext = os.path.splitext(basename)[0]
            save_path = os.path.join(cam_out_dir, f"{name_no_ext}.npz")
            if not os.path.exists(save_path):
                unprocessed_paths.append(p)
                
        if LIMIT:
            unprocessed_paths = unprocessed_paths[:LIMIT]
            
        if not unprocessed_paths:
            print(f"\n{cam_name}: All {len(img_paths)} images already processed.")
            continue
            
        print(f"\nProcessing {cam_name}: {len(unprocessed_paths)} images at {INPUT_WIDTH}x{INPUT_HEIGHT}...")
        
        dataset = ImageDataset(unprocessed_paths, INPUT_HEIGHT, INPUT_WIDTH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        processed_count = 0
        total = len(unprocessed_paths)
        
        for batch_paths, batch_tensors, batch_orig_w, batch_orig_h in dataloader:
            batch_tensors = batch_tensors.to(device)
            
            with torch.no_grad():
                features_dict = model.forward_features(batch_tensors)
                patch_tokens = features_dict["x_norm_patchtokens"]
            
            features_batch = patch_tokens.cpu().numpy()
            features_batch = features_batch.reshape(-1, patch_h, patch_w, feat_dim)
            
            for i, path in enumerate(batch_paths):
                basename = os.path.basename(path)
                name_no_ext = os.path.splitext(basename)[0]
                save_path = os.path.join(cam_out_dir, f"{name_no_ext}.npz")
                vis_path = os.path.join(cam_out_dir, f"{name_no_ext}_vis.png")
                
                features = features_batch[i]
                orig_w, orig_h = int(batch_orig_w[i]), int(batch_orig_h[i])
                
                np.savez_compressed(
                    save_path,
                    features=features.astype(np.float16),
                    orig_size=np.array([orig_h, orig_w]),
                    patch_size=np.array([14]),
                    input_size=np.array([INPUT_HEIGHT, INPUT_WIDTH]),
                )
                
                if not VISUALIZE_FIRST_ONLY or processed_count == 0:
                    vis = create_pca_visualization(features)
                    vis_upscaled = Image.fromarray(vis).resize((orig_w, orig_h), Image.BILINEAR)
                    vis_upscaled.save(vis_path)
                
                processed_count += 1
                if processed_count % min(10, BATCH_SIZE) == 0 or processed_count == total:
                    print(f"\r  Progress: [{processed_count}/{total}]", end="")
                    
        print() # New line after progress
        
    print(f"\nFeature extraction complete.")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
