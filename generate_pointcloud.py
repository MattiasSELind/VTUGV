
import os
import glob
import numpy as np
from PIL import Image
import itertools

# Configuration
DEPTH_DIR = r"data/depth"
IMAGE_DIR = r"data/images/multisense_left_image_rect_color"
SEMANTIC_DIR = r"data/segmentation/labels"
FEATURE_VIS_DIR = r"data/features/vis"
OUTPUT_DIR = r"data/pointclouds"

# Intrinsics (from data/calibration/multisense_intrinsics.txt)
FX = 477.6049499511719
FY = 477.6049499511719
CX = 499.5
CY = 252.0

# ADE20K Colormap for semantic segmentation (random but fixed 150 colors)
np.random.seed(42)
SEMANTIC_COLORMAP = np.random.randint(0, 255, (256, 3), dtype=np.uint8)

def save_ply(output_path, points, colors):
    """
    Save point cloud to PLY file.
    points: (N, 3) float32
    colors: (N, 3) uint8
    """
    # Header
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    # Combine points and colors
    data = np.hstack([points, colors.astype(np.float32)])
    
    # Save
    with open(output_path, "w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.4f %.4f %.4f %d %d %d")

def main():
    # Create output directories
    os.makedirs(os.path.join(OUTPUT_DIR, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "semantic"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "features"), exist_ok=True)

    # Get file lists
    # Assuming filenames match: frame_XXXXXX.png/npy/etc
    # Depth files are named depth_XXXXXX.npy
    # Image files are frame_XXXXXX.png
    
    depth_files = sorted(glob.glob(os.path.join(DEPTH_DIR, "depth_*.npy")))
    
    if not depth_files:
        print("No depth files found.")
        return
        
    print(f"Found {len(depth_files)} depth files. Processing...")

    for i, depth_path in enumerate(depth_files):
        # Extract ID
        basename = os.path.basename(depth_path)
        frame_id_str = basename.replace("depth_", "").replace(".npy", "")
        
        # Construct paths
        img_path = os.path.join(IMAGE_DIR, f"frame_{frame_id_str}.png")
        sem_path = os.path.join(SEMANTIC_DIR, f"frame_{frame_id_str}.png")
        feat_path = os.path.join(FEATURE_VIS_DIR, f"frame_{frame_id_str}_pca.png")
        
        # Check existence
        if not os.path.exists(img_path):
            print(f"Image not found for {frame_id_str}, skipping.")
            continue
            
        print(f"[{i+1}/{len(depth_files)}] Processing frame {frame_id_str}...")
        
        try:
            # Load Data
            depth = np.load(depth_path).astype(np.float32) # (H, W) or (1, H, W)
            if depth.ndim == 3:
                depth = depth.squeeze()
                
            rgb_img = Image.open(img_path).convert("RGB")
            rgb_arr = np.array(rgb_img) # (H, W, 3)
            
            # Semantic
            if os.path.exists(sem_path):
                sem_img = Image.open(sem_path)
                sem_arr = np.array(sem_img) # (H, W)
                # Apply colormap
                sem_color = SEMANTIC_COLORMAP[sem_arr] # (H, W, 3)
            else:
                sem_color = None
                print(f"Warning: Semantic mask not found for {frame_id_str}")

            # Features
            if os.path.exists(feat_path):
                feat_img = Image.open(feat_path).convert("RGB")
                # Resize if needed (should match, but good to ensure)
                if feat_img.size != rgb_img.size:
                    feat_img = feat_img.resize(rgb_img.size, Image.NEAREST)
                feat_color = np.array(feat_img)
            else:
                feat_color = None
                print(f"Warning: Feature visualization not found for {frame_id_str}")

            # Back-projection
            H, W = depth.shape
            
            # Meshgrid
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            
            # Mask valid depth (filter out 0 or huge values)
            valid_mask = (depth > 0.1) & (depth < 100.0)
            
            # Flatten
            u_flat = u[valid_mask]
            v_flat = v[valid_mask]
            z_flat = depth[valid_mask]
            
            # Project
            x_flat = (u_flat - CX) * z_flat / FX
            y_flat = (v_flat - CY) * z_flat / FY
            
            points = np.stack([x_flat, y_flat, z_flat], axis=1) # (N, 3)
            
            # --- Save RGB Point Cloud ---
            rgb_flat = rgb_arr[valid_mask]
            save_ply(os.path.join(OUTPUT_DIR, "rgb", f"cloud_{frame_id_str}.ply"), points, rgb_flat)
            
            # --- Save Semantic Point Cloud ---
            if sem_color is not None:
                sem_flat = sem_color[valid_mask]
                save_ply(os.path.join(OUTPUT_DIR, "semantic", f"cloud_{frame_id_str}.ply"), points, sem_flat)
                
            # --- Save Feature Point Cloud ---
            if feat_color is not None:
                feat_flat = feat_color[valid_mask]
                save_ply(os.path.join(OUTPUT_DIR, "features", f"cloud_{frame_id_str}.ply"), points, feat_flat)
                
        except Exception as e:
            print(f"Error processing frame {frame_id_str}: {e}")
            import traceback
            traceback.print_exc()

    print("Point cloud generation complete.")

if __name__ == "__main__":
    main()
