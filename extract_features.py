
import os
import torch
import glob
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt

# Configuration
IMAGE_DIR = os.path.abspath(r"data/images/multisense_left_image_rect_color")
FEATURE_DIR = r"data/features"
VIS_DIR = r"data/features/vis"
MODEL_NAME = "facebook/dinov2-large"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Ensure output directories exist
    os.makedirs(FEATURE_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    # Load Model
    print(f"Loading model {MODEL_NAME}...")
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Process Images
    if os.path.exists(IMAGE_DIR):
        print(f"Directory {IMAGE_DIR} exists.")
        print(f"Contents: {os.listdir(IMAGE_DIR)[:5]}")
    else:
        print(f"Directory {IMAGE_DIR} DOES NOT EXIST.")

    search_path = os.path.join(IMAGE_DIR, "*.png")
    print(f"Searching for images in: {search_path}")
    image_paths = sorted(glob.glob(search_path))
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}")
        print(f"Current working directory: {os.getcwd()}")
        return
        
    print(f"Found {len(image_paths)} images. Processing...")

    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        feature_save_path = os.path.join(FEATURE_DIR, filename.replace(".png", "_features.pt"))
        vis_save_path = os.path.join(VIS_DIR, filename.replace(".png", "_pca.png"))

        # Force re-processing for fine-grained features request
        # if os.path.exists(feature_save_path) and os.path.exists(vis_save_path):
        #      print(f"[{i+1}/{len(image_paths)}] Skipping {filename} (already exists)")
        #      continue

        print(f"[{i+1}/{len(image_paths)}] Processing {filename}...")

        try:
            image = Image.open(img_path).convert("RGB")
            W, H = image.size
            
            # Resize to be multiple of patch_size (14)
            patch_size = 14
            new_W = int(round(W / patch_size) * patch_size)
            new_H = int(round(H / patch_size) * patch_size)
            
            # Resize image if needed
            if (new_W, new_H) != (W, H):
                image = image.resize((new_W, new_H), resample=Image.BICUBIC)
                
            # Prepare inputs with NO resizing from processor to keep high res
            inputs = processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract Last Hidden State
            last_hidden_state = outputs.last_hidden_state
            
            # Remove CLS token (index 0)
            patch_features = last_hidden_state[:, 1:, :] # (1, num_patches, 1024)
            
            # Calculate grid size
            num_patches = patch_features.shape[1]
            H_input, W_input = inputs['pixel_values'].shape[-2:]
            patch_size = 14 # DINOv2 standard
            H_patches = H_input // patch_size
            W_patches = W_input // patch_size
            
            if H_patches * W_patches != num_patches:
                 print(f"Warning: Patch mismatch for {filename}. Expected {H_patches*W_patches}, got {num_patches}. Skipping visualization.")
                 torch.save(patch_features.cpu().half(), feature_save_path)
                 continue

            # Reshape features to spatial map
            spatial_features = patch_features.reshape(1, H_patches, W_patches, -1) # (1, H, W, D)
            
            # Save Raw Features
            torch.save(spatial_features.cpu().half(), feature_save_path)

            # --- Visualization (PCA using Torch) ---
            # Flatten: (H*W, D)
            flat_features = patch_features.squeeze(0) # (N, D)
            
            # Center data
            mean = torch.mean(flat_features, dim=0)
            centered_features = flat_features - mean
            
            # Generic PCA using SVD or pca_lowrank
            # For visualization (3 components), pca_lowrank is efficient
            try:
                # torch.pca_lowrank returns (U, S, V)
                # U is (N, q), S is (q,), V is (D, q)
                # projected = centered * V
                _, _, V = torch.pca_lowrank(centered_features, q=3, center=False, niter=2)
                pca_features = torch.matmul(centered_features, V) # (N, 3)
            except Exception as pca_err:
                 print(f"PCA failed: {pca_err}")
                 continue

            # Min-Max Normalize to 0-255
            pca_min = pca_features.min(dim=0).values
            pca_max = pca_features.max(dim=0).values
            pca_features_norm = (pca_features - pca_min) / (pca_max - pca_min + 1e-6)
            pca_features_uint8 = (pca_features_norm * 255).cpu().numpy().astype(np.uint8)
            
            # Reshape to image
            pca_img_array = pca_features_uint8.reshape(H_patches, W_patches, 3)
            
            # Resize visualization
            vis_img = Image.fromarray(pca_img_array)
            vis_img = vis_img.resize((image.width, image.height), resample=Image.NEAREST)
            vis_img.save(vis_save_path)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

    print("Processing complete.")

if __name__ == "__main__":
    main()
