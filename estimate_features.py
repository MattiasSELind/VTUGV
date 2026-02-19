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

# Configuration
HOME = os.path.expanduser("~")
DATASET_DIR = os.path.join(HOME, "Downloads", "the_great_outdoors_data")
IMAGE_DIR = DATASET_DIR
OUTPUT_DIR = os.path.join(HOME, "Downloads", "the_great_outdoors_data_features")

CAMERAS = ["front_left", "front_right", "rear_center"]

# Limit for testing — set to None for full dataset
LIMIT = 20

# DINOv2 model size: "small" (384-dim, fast) or "base" (768-dim, richer)
MODEL_SIZE = "small"

# Input resolution for DINOv2 (must be multiple of 14 for ViT patch size)
# Higher = more spatial detail but slower. 518 gives 37x37 patches.
INPUT_SIZE = 518


def load_dinov2(model_size="small", device="cpu"):
    """Load DINOv2 model from torch hub."""
    model_name = f"dinov2_vit{model_size[0]}14"  # e.g. dinov2_vits14
    print(f"Loading DINOv2: {model_name}")
    
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.to(device)
    model.eval()
    
    # Feature dimension
    feat_dim = {"small": 384, "base": 768, "large": 1024, "giant": 1536}[model_size]
    print(f"  Feature dimension: {feat_dim}")
    print(f"  Input size: {INPUT_SIZE}x{INPUT_SIZE}")
    
    num_patches = INPUT_SIZE // 14
    print(f"  Patch grid: {num_patches}x{num_patches} = {num_patches**2} patches")
    
    return model, feat_dim


def extract_features(model, image, device, input_size=518):
    """
    Extract dense patch features from an image using DINOv2.
    
    Returns:
        features: (num_patches_h, num_patches_w, feat_dim) numpy array
        patch_size: size of each patch in pixels at input resolution
    """
    # Preprocess: resize + normalize (ImageNet stats, matching DINOv2 training)
    img_resized = image.resize((input_size, input_size), Image.BILINEAR)
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_arr - mean) / std
    
    # (H, W, 3) -> (1, 3, H, W)
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        # Get patch tokens (excluding CLS token)
        features_dict = model.forward_features(img_tensor)
        patch_tokens = features_dict["x_norm_patchtokens"]  # (1, num_patches, feat_dim)
    
    # Reshape to spatial grid
    num_patches = input_size // 14
    features = patch_tokens.squeeze(0).cpu().numpy()  # (num_patches^2, feat_dim)
    features = features.reshape(num_patches, num_patches, -1)  # (H_p, W_p, feat_dim)
    
    return features


def create_pca_visualization(features, n_components=3):
    """
    Create a false-color RGB visualization of features using PCA.
    
    Args:
        features: (H, W, D) feature map
    Returns:
        vis: (H, W, 3) uint8 image
    """
    H, W, D = features.shape
    flat = features.reshape(-1, D)
    
    # Center
    mean = flat.mean(axis=0)
    centered = flat - mean
    
    # PCA via SVD (only need top 3 components)
    # For efficiency, use covariance method when D < N
    if D < flat.shape[0]:
        cov = centered.T @ centered / flat.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Take top 3 (eigenvalues returned in ascending order)
        top3 = eigenvectors[:, -n_components:][:, ::-1]
    else:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        top3 = Vt[:n_components].T
    
    projected = centered @ top3  # (N, 3)
    
    # Normalize each channel to [0, 255]
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
    num_patches = INPUT_SIZE // 14
    
    # Process each camera
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
        
        if LIMIT:
            img_paths = img_paths[:LIMIT]
        
        print(f"\nProcessing {cam_name}: {len(img_paths)} images...")
        
        for i, img_path in enumerate(img_paths):
            basename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(basename)[0]
            save_path = os.path.join(cam_out_dir, f"{name_no_ext}.npz")
            vis_path = os.path.join(cam_out_dir, f"{name_no_ext}_vis.png")
            
            if os.path.exists(save_path):
                continue
            
            print(f"  [{i+1}/{len(img_paths)}] {cam_name}/{basename}", end="")
            
            try:
                image = Image.open(img_path).convert("RGB")
                orig_w, orig_h = image.size
                
                # Extract features
                features = extract_features(model, image, device, INPUT_SIZE)
                # features shape: (37, 37, 384) for small model @ 518px
                
                # Save compressed
                np.savez_compressed(
                    save_path,
                    features=features.astype(np.float16),  # Half precision to save space
                    orig_size=np.array([orig_h, orig_w]),
                    patch_size=np.array([14]),
                    input_size=np.array([INPUT_SIZE]),
                )
                
                # PCA visualization
                vis = create_pca_visualization(features)
                vis_upscaled = Image.fromarray(vis).resize((orig_w, orig_h), Image.BILINEAR)
                vis_upscaled.save(vis_path)
                
                # File size info for first image
                if i == 0:
                    fsize = os.path.getsize(save_path) / 1024
                    print(f" | {features.shape} | {fsize:.0f}KB")
                else:
                    print(" ✓")
                
            except Exception as e:
                print(f" ERROR: {e}")
    
    print(f"\nFeature extraction complete.")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Feature shape per image: ({num_patches}, {num_patches}, {feat_dim})")
    print(f"Model: DINOv2-{MODEL_SIZE} @ {INPUT_SIZE}px")


if __name__ == "__main__":
    main()
