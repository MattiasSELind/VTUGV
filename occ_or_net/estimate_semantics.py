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
DATASET_DIR = os.path.join(HOME, "Downloads", "Data_Outdoors", "images")
IMAGE_DIR = DATASET_DIR
OUTPUT_DIR = os.path.join(HOME, "Downloads", "the_great_outdoors_data_semantics")

CAMERAS = [ "front_right", "rear_center"]

# Limit for testing — set to None for full dataset
LIMIT = 7252

# Optimization Configuration
BATCH_SIZE = 8
VISUALIZE_FIRST_ONLY = True

# ====================================================================
# CUSTOM OFF-ROAD CLASS DEFINITIONS
# ====================================================================
CUSTOM_CLASSES = [
    "dirt road",     # 0
    "grass",         # 1
    "tree",          # 2
    "bush",          # 3
    "rock",          # 4
    "mud",           # 5
    "water",         # 6
    "sky",           # 7
    "vehicle",       # 8
    "person",        # 9
    "building",      # 10
    "fence",         # 11
    "terrain",       # 12
]

CLIP_PROMPTS = [
    "an unpaved dirt road or gravel trail on the ground",          # 0 dirt road
    "green grass growing on the ground",                           # 1 grass
    "a tall tree with trunk and branches and leaves",              # 2 tree
    "a bush or shrub or low undergrowth vegetation",               # 3 bush
    "a rock or stone or boulder on the ground",                    # 4 rock
    "wet muddy ground or puddle of mud",                           # 5 mud
    "a body of water such as a stream or river or puddle",        # 6 water
    "the sky with clouds above the horizon",                       # 7 sky
    "a car or truck or vehicle driving on a road",                 # 8 vehicle
    "a human person standing or walking upright",                  # 9 person
    "a man-made building or structure or house",                   # 10 building
    "a fence or railing or barrier",                               # 11 fence
    "flat bare ground or sandy terrain without vegetation",        # 12 terrain
]

CUSTOM_PALETTE = np.array([
    [160, 120, 80],    # 0  dirt road  -> 🟤 brown
    [124, 252, 0],     # 1  grass      -> 🟢 lawn green
    [34, 139, 34],     # 2  tree       -> 🌲 forest green
    [0, 180, 0],       # 3  bush       -> 🟢 green
    [160, 160, 160],   # 4  rock       -> ⬜ gray
    [100, 70, 40],     # 5  mud        -> 🟤 dark brown
    [0, 100, 255],     # 6  water      -> 🔵 blue
    [135, 206, 250],   # 7  sky        -> 🔵 light blue
    [0, 0, 180],       # 8  vehicle    -> 🔵 dark blue
    [255, 0, 0],       # 9  person     -> 🔴 red
    [70, 70, 70],      # 10 building   -> ⬛ dark gray
    [190, 153, 153],   # 11 fence      -> ⬜ blush
    [210, 180, 140],   # 12 terrain    -> 🟡 tan
], dtype=np.uint8)

NUM_CLASSES = len(CUSTOM_CLASSES)

def segment_image_batch(model, processor, images, class_labels, device):
    """
    Run CLIPSeg on a batch of images with the given class labels.
    Returns list of (H, W) uint8 arrays of class indices.
    """
    texts = class_labels * len(images)
    repeated_images = [img for img in images for _ in range(len(class_labels))]
    
    inputs = processor(
        text=texts,
        images=repeated_images,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits 
    logits = logits.view(len(images), len(class_labels), logits.shape[-2], logits.shape[-1])
    
    seg_maps = []
    for i, image in enumerate(images):
        orig_w, orig_h = image.size
        img_logits = logits[i]
        
        if img_logits.dim() == 2:
            img_logits = img_logits.unsqueeze(0)
            
        logits_up = torch.nn.functional.interpolate(
            img_logits.unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        
        seg_map = logits_up.argmax(dim=0).cpu().numpy().astype(np.uint8)
        seg_maps.append(seg_map)
        
    return seg_maps

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
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
        
        unprocessed_paths = []
        for p in img_paths:
            basename = os.path.basename(p)
            name_no_ext = os.path.splitext(basename)[0]
            save_path = os.path.join(cam_out_dir, f"{name_no_ext}.npy")
            if not os.path.exists(save_path):
                unprocessed_paths.append(p)
        
        if LIMIT:
            unprocessed_paths = unprocessed_paths[:LIMIT]
            
        if not unprocessed_paths:
            print(f"\n{cam_name}: All {len(img_paths)} images already processed.")
            continue
        
        print(f"\nProcessing {cam_name}: {len(unprocessed_paths)} images...")
        
        processed_count = 0
        total = len(unprocessed_paths)
        
        for batch_start in range(0, total, BATCH_SIZE):
            batch_paths = unprocessed_paths[batch_start:batch_start + BATCH_SIZE]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            seg_maps = segment_image_batch(model, processor, batch_images, CLIP_PROMPTS, device)
            
            for i, (path, image) in enumerate(zip(batch_paths, batch_images)):
                basename = os.path.basename(path)
                name_no_ext = os.path.splitext(basename)[0]
                save_path = os.path.join(cam_out_dir, f"{name_no_ext}.npy")
                vis_path = os.path.join(cam_out_dir, f"{name_no_ext}_vis.png")
                
                seg_map = seg_maps[i]
                np.save(save_path, seg_map)
                
                if not VISUALIZE_FIRST_ONLY or processed_count == 0:
                    vis = CUSTOM_PALETTE[seg_map]
                    orig_arr = np.array(image)
                    if orig_arr.shape[:2] != vis.shape[:2]:
                        image_resized = image.resize((vis.shape[1], vis.shape[0]))
                        orig_arr = np.array(image_resized)
                    
                    blended = (0.4 * orig_arr + 0.6 * vis).astype(np.uint8)
                    Image.fromarray(blended).save(vis_path)
                
                processed_count += 1
                if processed_count % min(10, BATCH_SIZE) == 0 or processed_count == total:
                    print(f"\r  Progress: [{processed_count}/{total}]", end="")
                    
        print() # New line after progress
        
    print("\nSemantic segmentation complete.")

if __name__ == "__main__":
    main()
