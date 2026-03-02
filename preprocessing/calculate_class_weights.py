import os
import glob
import numpy as np
import torch

# Configuration matching your estimate_semantics.py
HOME = os.path.expanduser("~")
OUTPUT_DIR = os.path.join(HOME, "Downloads", "Data_Outdoors", "semantics")
CAMERAS = ["front_left", "rear_center"]
NUM_CLASSES = 13

def calculate_weights():
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_pixels = 0

    print("Counting pixels across all semantic maps...")
    for cam_name in CAMERAS:
        cam_dir = os.path.join(OUTPUT_DIR, cam_name)
        if not os.path.exists(cam_dir):
            continue
        
        npy_files = glob.glob(os.path.join(cam_dir, "*.npy"))
        
        for npy_path in npy_files:
            seg_map = np.load(npy_path)
            # Count occurrences of each class in this image
            counts = np.bincount(seg_map.flatten(), minlength=NUM_CLASSES)
            class_counts += (counts[:NUM_CLASSES] if len(counts) > NUM_CLASSES else np.pad(counts, (0, max(0, NUM_CLASSES - len(counts)))))
            total_pixels += seg_map.size

    if total_pixels == 0:
        print("No semantic maps found. Weights cannot be calculated.")
        return

    # 1. Calculate the frequency of each class
    # Add a tiny epsilon to prevent division by zero for classes that never appear
    epsilon = 1e-8
    class_frequencies = class_counts / (total_pixels + epsilon)

    # 2. Median Frequency Balancing
    # We ignore classes with 0 count when finding the median to avoid skewing it
    valid_frequencies = class_frequencies[class_counts > 0]
    
    if len(valid_frequencies) == 0:
        print("No pixels found. Exiting.")
        return
        
    median_freq = np.median(valid_frequencies)

    # Weight = median_frequency / class_frequency
    class_weights = median_freq / (class_frequencies + epsilon)

    # Cap weights to prevent exploding gradients on extremely rare classes (optional but recommended)
    max_weight_cap = 50.0 
    class_weights = np.clip(class_weights, a_min=None, a_max=max_weight_cap)

    # Set weights to 0 for classes that literally don't exist in the dataset
    class_weights[class_counts == 0] = 0.0 

    print("\nCalculated Class Weights:")
    for i, w in enumerate(class_weights):
        print(f"Class {i}: {w:.4f} (Pixels: {class_counts[i]})")

    # Save the weights so your training script can load them
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    save_path = os.path.join(OUTPUT_DIR, "class_weights.pt")
    
    # Ensure directory exists just in case
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    torch.save(weights_tensor, save_path)
    print(f"\nSaved weights to: {save_path}")

if __name__ == "__main__":
    calculate_weights()
