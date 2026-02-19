"""
estimate_semantics.py
Zero-shot semantic segmentation using CLIPSeg with custom off-road labels.
No dataset-specific training needed — we define our own class labels!
Saves per-pixel class index maps as .npy (uint8) and visualization PNGs.
"""

import os
import glob
import numpy as np
from PIL import Image

# Set HF cache before importing transformers
os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), "Downloads", "hf_cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Configuration
HOME = os.path.expanduser("~")
DATASET_DIR = os.path.join(HOME, "Downloads", "the_great_outdoors_data")
IMAGE_DIR = DATASET_DIR
OUTPUT_DIR = os.path.join(HOME, "Downloads", "the_great_outdoors_data_semantics")

CAMERAS = ["front_left", "front_right", "rear_center"]

# Limit for testing — set to None for full dataset
LIMIT = 20

# ====================================================================
# CUSTOM OFF-ROAD CLASS DEFINITIONS
# Add, remove, or rename these to match your dataset's environment.
# CLIPSeg uses CLIP text embeddings, so natural language works well.
# ====================================================================
CUSTOM_CLASSES = [
    "dirt road",          # 0  - unpaved driving surface, gravel path
    "grass",              # 1  - ground-level vegetation, lawn
    "tree",               # 2  - trees, tree trunks, branches
    "bush",               # 3  - shrubs, undergrowth, low vegetation
    "rock",               # 4  - stones, boulders, rocky surface
    "mud",                # 5  - wet ground, muddy surface
    "water",              # 6  - puddles, streams, lakes
    "sky",                # 7  - sky, clouds
    "vehicle",            # 8  - cars, trucks, ATVs
    "person",             # 9  - people, pedestrians
    "building",           # 10 - structures, houses, sheds
    "fence",              # 11 - fences, barriers, railings
    "terrain",            # 12 - general undifferentiated ground
]

# 0: dirt road       🟤    7: sky          🔵
# 1: grass           🟢    8: vehicle      🔵
# 2: tree            🟢    9: person       🔴
# 3: bush            🟢   10: building     ⬛
# 4: rock            ⬜   11: fence        ⬜
# 5: mud             🟤   12: terrain      🟡
# 6: water           🔵

# Colors for each class (for visualization and 3D point clouds)
CUSTOM_PALETTE = np.array([
    [160, 120, 80],    # 0  dirt road  -> brown
    [124, 252, 0],     # 1  grass      -> lawn green
    [34, 139, 34],     # 2  tree       -> forest green
    [0, 180, 0],       # 3  bush       -> green
    [160, 160, 160],   # 4  rock       -> gray
    [100, 70, 40],     # 5  mud        -> dark brown
    [0, 100, 255],     # 6  water      -> blue
    [135, 206, 250],   # 7  sky        -> light blue
    [0, 0, 180],       # 8  vehicle    -> dark blue
    [255, 0, 0],       # 9  person     -> red
    [70, 70, 70],      # 10 building   -> dark gray
    [190, 153, 153],   # 11 fence      -> blush
    [210, 180, 140],   # 12 terrain    -> tan
], dtype=np.uint8)

NUM_CLASSES = len(CUSTOM_CLASSES)


def segment_image(model, processor, image, class_labels, device):
    """
    Run CLIPSeg on a single image with the given class labels.
    Returns (H, W) uint8 array of class indices.
    """
    # CLIPSeg expects one text prompt per class
    inputs = processor(
        text=class_labels,
        images=[image] * len(class_labels),
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # outputs.logits: (num_classes, H_small, W_small)
    logits = outputs.logits  # (num_classes, H/?, W/?)
    
    # Upsample to original image size
    orig_h, orig_w = image.size[1], image.size[0]
    
    if logits.dim() == 2:
        # Single class case — shouldn't happen but handle
        logits = logits.unsqueeze(0)
    
    # (num_classes, H, W) -> upsample
    logits_up = torch.nn.functional.interpolate(
        logits.unsqueeze(0) if logits.dim() == 3 else logits,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False
    )
    
    # Squeeze batch dim if present
    if logits_up.dim() == 4:
        logits_up = logits_up.squeeze(0)  # (num_classes, H, W)
    
    # Argmax over classes
    seg_map = logits_up.argmax(dim=0).cpu().numpy().astype(np.uint8)
    
    return seg_map


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIPSeg
    model_name = "CIDAS/clipseg-rd64-refined"
    print(f"Loading model: {model_name}")
    
    try:
        processor = CLIPSegProcessor.from_pretrained(model_name)
        model = CLIPSegForImageSegmentation.from_pretrained(model_name)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    print(f"\nCustom classes ({NUM_CLASSES}):")
    for i, cls in enumerate(CUSTOM_CLASSES):
        r, g, b = CUSTOM_PALETTE[i]
        print(f"  {i:2d}: {cls:20s}  RGB({r:3d},{g:3d},{b:3d})")
    
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
            save_path = os.path.join(cam_out_dir, f"{name_no_ext}.npy")
            vis_path = os.path.join(cam_out_dir, f"{name_no_ext}_vis.png")
            
            if os.path.exists(save_path):
                continue
            
            print(f"  [{i+1}/{len(img_paths)}] {cam_name}/{basename}", end="")
            
            try:
                image = Image.open(img_path).convert("RGB")
                
                seg_map = segment_image(model, processor, image, CUSTOM_CLASSES, device)
                
                # Save class map
                np.save(save_path, seg_map)
                
                # Save visualization (overlay)
                vis = CUSTOM_PALETTE[seg_map]  # (H, W, 3)
                
                # Blend with original for better visualization
                orig_arr = np.array(image)
                if orig_arr.shape[:2] != vis.shape[:2]:
                    image_resized = image.resize((vis.shape[1], vis.shape[0]))
                    orig_arr = np.array(image_resized)
                
                blended = (0.4 * orig_arr + 0.6 * vis).astype(np.uint8)
                Image.fromarray(blended).save(vis_path)
                
                # Print detected classes for first image per camera
                if i == 0:
                    unique, counts = np.unique(seg_map, return_counts=True)
                    total = seg_map.size
                    print()
                    for cls_id, count in zip(unique, counts):
                        pct = 100 * count / total
                        if pct > 1.0:  # Only show classes > 1%
                            print(f"      {CUSTOM_CLASSES[cls_id]:20s}: {pct:.1f}%")
                else:
                    print(" ✓")
                
            except Exception as e:
                print(f" ERROR: {e}")
    
    print("\nSemantic segmentation complete.")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
