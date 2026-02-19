"""
estimate_semantics.py
Run semantic segmentation on Great Outdoors images using SegFormer (HuggingFace).
Uses ADE20K dataset (150 classes) — suitable for off-road/outdoor scenes.
Saves per-pixel class index maps as .npy files (uint8) and visualization PNGs.
"""

import os
import glob
import numpy as np
from PIL import Image

# Set HF cache before importing transformers
os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), "Downloads", "hf_cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Configuration
HOME = os.path.expanduser("~")
DATASET_DIR = os.path.join(HOME, "Downloads", "the_great_outdoors_data")
IMAGE_DIR = DATASET_DIR
OUTPUT_DIR = os.path.join(HOME, "Downloads", "the_great_outdoors_data_semantics")

CAMERAS = ["front_left", "front_right", "rear_center"]

# Limit for testing — set to None for full dataset
LIMIT = 20

# ADE20K 150 class names (index 0-149)
ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree",
    "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf",
    "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock",
    "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers",
    "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway",
    "case", "pool table", "pillow", "screen door", "stairway",
    "river", "bridge", "bookcase", "blind", "coffee table",
    "toilet", "flower", "book", "hill", "bench",
    "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel",
    "bus", "towel", "light", "truck", "tower",
    "chandelier", "awning", "streetlight", "booth", "television",
    "airplane", "dirt track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
    "stool", "barrel", "basket", "waterfall", "tent",
    "bag", "minibike", "cradle", "oven", "ball",
    "food", "step", "tank", "trade name", "microwave",
    "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce",
    "vase", "traffic light", "tray", "ashcan", "fan",
    "pier", "crt screen", "plate", "monitor", "bulletin board",
    "shower", "radiator", "glass", "clock", "flag",
]

# ADE20K color palette (150 colors — standard visualization)
# Generated to be visually distinct per class
def _generate_ade20k_palette():
    """Generate a deterministic 150-color palette for ADE20K."""
    palette = np.zeros((150, 3), dtype=np.uint8)
    for i in range(150):
        r, g, b = 0, 0, 0
        idx = i + 1  # 1-indexed for bit extraction
        for j in range(8):
            r = r | ((idx >> 0) & 1) << (7 - j)
            g = g | ((idx >> 1) & 1) << (7 - j)
            b = b | ((idx >> 2) & 1) << (7 - j)
            idx >>= 3
        palette[i] = [r, g, b]
    return palette

ADE20K_PALETTE = _generate_ade20k_palette()

# Override specific classes with more intuitive colors for off-road
ADE20K_PALETTE[4]  = [34, 139, 34]    # tree -> forest green
ADE20K_PALETTE[9]  = [124, 252, 0]    # grass -> lawn green
ADE20K_PALETTE[13] = [139, 90, 43]    # earth -> brown
ADE20K_PALETTE[6]  = [128, 128, 128]  # road -> gray
ADE20K_PALETTE[21] = [0, 100, 255]    # water -> blue
ADE20K_PALETTE[34] = [160, 160, 160]  # rock -> light gray
ADE20K_PALETTE[17] = [0, 200, 0]      # plant -> green
ADE20K_PALETTE[46] = [238, 214, 175]  # sand -> tan
ADE20K_PALETTE[2]  = [135, 206, 250]  # sky -> light blue
ADE20K_PALETTE[52] = [210, 180, 140]  # path -> tan
ADE20K_PALETTE[91] = [160, 120, 80]   # dirt track -> brown
ADE20K_PALETTE[16] = [100, 130, 100]  # mountain -> muted green
ADE20K_PALETTE[29] = [180, 220, 80]   # field -> yellow-green
ADE20K_PALETTE[69] = [90, 160, 90]    # hill -> dark green
ADE20K_PALETTE[12] = [255, 0, 0]      # person -> red
ADE20K_PALETTE[20] = [0, 0, 180]      # car -> dark blue
ADE20K_PALETTE[31] = [190, 153, 153]  # fence -> blush


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # SegFormer-b2 on ADE20K — good balance of speed and accuracy
    # b0 is fastest, b5 is most accurate
    model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
    print(f"Loading model: {model_name}")
    
    try:
        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
        print(f"Classes: {len(ADE20K_CLASSES)} (ADE20K)")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Process each camera
    for cam_name in CAMERAS:
        cam_in_dir = os.path.join(IMAGE_DIR, cam_name)
        cam_out_dir = os.path.join(OUTPUT_DIR, cam_name)
        
        if not os.path.isdir(cam_in_dir):
            print(f"Camera folder not found: {cam_in_dir}")
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
            
            print(f"  [{i+1}/{len(img_paths)}] {cam_name}/{basename}")
            
            try:
                image = Image.open(img_path).convert("RGB")
                orig_size = image.size  # (W, H)
                
                # Preprocess
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                # Inference
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Get class predictions
                logits = outputs.logits  # (1, 150, H/4, W/4)
                
                # Upsample to original image size
                upsampled = torch.nn.functional.interpolate(
                    logits,
                    size=(orig_size[1], orig_size[0]),  # (H, W)
                    mode="bilinear",
                    align_corners=False
                )
                
                seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
                
                # Save class map
                np.save(save_path, seg_map)
                
                # Save visualization
                vis = ADE20K_PALETTE[seg_map]  # (H, W, 3)
                Image.fromarray(vis).save(vis_path)
                
                # Print detected classes for first image
                if i == 0:
                    unique_classes = np.unique(seg_map)
                    print(f"    Detected classes: {[ADE20K_CLASSES[c] for c in unique_classes]}")
                
            except Exception as e:
                print(f"  Error: {e}")
    
    print("\nSemantic segmentation complete.")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nKey off-road classes to look for:")
    for idx in [4, 9, 13, 6, 21, 34, 17, 46, 91, 29, 52, 69, 16]:
        print(f"  {idx:3d}: {ADE20K_CLASSES[idx]}")


if __name__ == "__main__":
    main()
