
import os
import glob
# Set cache BEFORE importing transformers
os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), "Downloads", "hf_cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import torch
import numpy as np
from PIL import Image
from transformers import pipeline

# Configuration
# Points to the specific dataset location
START_DIR = "C:\\Users\\maxlars\\Downloads\\the_great_outdoors_data"

if not os.path.exists(START_DIR):
    START_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "the_great_outdoors_data")


IMAGE_DIR = START_DIR
OUTPUT_DIR = START_DIR + "_depth_estimated" # Keep separate from provided depth


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    pipe = None
    last_error = None
    
    # Try multiple popular checkpoints for depth
    checkpoints = [
        "depth-anything/Depth-Anything-V2-Small-hf",
        "LiheYoung/depth-anything-small-hf",
        "Intel/dpt-large"
    ]
    
    for ckpt in checkpoints:
        print(f"Attempting to load model: {ckpt}")
        try:
            pipe = pipeline(task="depth-estimation", model=ckpt, device=0 if device == "cuda" else -1)
            print(f"Successfully loaded {ckpt}")
            break
        except Exception as e:
            print(f"Failed to load {ckpt}: {e}")
            last_error = e
            
    if pipe is None:
        print(f"Could not load any depth estimation models. Last error: {last_error}")
        return

    # Process all subdirectories (cam1, cam2, cam3, etc.)
    if not os.path.exists(IMAGE_DIR):
        print(f"Input directory not found: {IMAGE_DIR}")
        print("Please ensure the Great Outdoors dataset is in your Downloads folder.")
        # Create empty folders for testing logic if needed, but for now just exit
        return

    # Process specific subdirectories
    target_cams = ["front_left", "front_right", "rear_center"]
    cam_folders = []
    
    entries = os.listdir(IMAGE_DIR)
    for cam in target_cams:
        if cam in entries and os.path.isdir(os.path.join(IMAGE_DIR, cam)):
            cam_folders.append(cam)
            
    if not cam_folders:
        print(f"No valid camera folders found in {IMAGE_DIR}. Looking for: {target_cams}")
        # Fallback logic preserved just in case
        # Fallback: check if images are directly in root
        if glob.glob(os.path.join(IMAGE_DIR, "*.png")) or glob.glob(os.path.join(IMAGE_DIR, "*.jpg")):
            print("Found images in root directory. Treating as single camera 'cam0'.")
            cam_folders = ["."]
        else:
            print(f"No subdirectories found in {IMAGE_DIR}. Expecting folders like 'cam1', 'cam2'...")
            return

    print(f"Found camera folders: {cam_folders}")

    for cam_name in cam_folders:
        if cam_name == ".":
            cam_in_dir = IMAGE_DIR
            cam_out_dir = OUTPUT_DIR
        else:
            cam_in_dir = os.path.join(IMAGE_DIR, cam_name)
            cam_out_dir = os.path.join(OUTPUT_DIR, cam_name)
        
        os.makedirs(cam_out_dir, exist_ok=True)
        
        # Gather images (recursive search or flat?)
        # Let's assume flat list inside cam folder
        img_paths = sorted(glob.glob(os.path.join(cam_in_dir, "*.png")) + glob.glob(os.path.join(cam_in_dir, "*.jpg")))
        
        if not img_paths:
             # Maybe images are in an 'images' subfolder?
             img_paths = sorted(glob.glob(os.path.join(cam_in_dir, "images", "*.png")) + glob.glob(os.path.join(cam_in_dir, "images", "*.jpg")))
             if img_paths:

                 print(f"Found images in 'images' subfolder for {cam_name}")
        
        print(f"Processing {cam_name}: {len(img_paths)} images...")

        # Processing all found images
        # LIMIT = 100 
        # img_paths = img_paths[:LIMIT]
        # print(f"Limiting to first {LIMIT} images.")
        print(f"Processing all {len(img_paths)} images.")
        
        for i, img_path in enumerate(img_paths):

            basename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(basename)[0]
            save_path = os.path.join(cam_out_dir, f"{name_no_ext}.npy") # Save as raw float32
            vis_path = os.path.join(cam_out_dir, f"{name_no_ext}_vis.png")
            
            if os.path.exists(save_path):
                continue
                
            print(f"[{i+1}/{len(img_paths)}] {cam_name}/{basename}")
            
            try:
                image = Image.open(img_path).convert("RGB")
                
                # Inference
                prediction = pipe(image)
                depth_map_pil = prediction["depth"] # PIL Image
                # prediction['predicted_depth'] is tensor (metric or inverse depth)
                # depth_anything usually returns relative depth. 
                # Let's trust the PIL output for now as 'relative depth map'.
                
                depth_np = np.array(depth_map_pil).astype(np.float32)
                
                # Normalize? Depth Anything usually outputs in large range.
                # If we treat it as metric depth directly, we might need a scale factor.
                # For now, save raw and we can tune scale in fusion.
                
                np.save(save_path, depth_np)
                
                # Save visualization
                depth_map_pil.save(vis_path)
                
            except Exception as e:
                print(f"Error processing {basename}: {e}")

    print("Depth estimation complete.")

if __name__ == "__main__":
    main()
