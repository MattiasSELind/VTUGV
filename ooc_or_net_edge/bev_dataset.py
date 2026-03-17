"""
bev_dataset.py
Simplified PyTorch Dataset for BEV prediction training.
Loads RGB images and pre-generated BEV pseudo-GT targets (semantics, height, slope).
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class BEVDataset(Dataset):
    """
    Dataset that pairs RGB camera images with pre-generated BEV targets.

    Directory layout expected:
        data_root/
            pylon_camera_node/0000/*.png          (RGB images)
            bev_targets/0000/*.npz                (BEV targets)
            camera_intrinsics/left_cam_intrinsic.txt

    Each .npz contains:
        bev_semantics  [100,100] uint8
        bev_height     [100,100] float32
        bev_slope      [100,100] float32
        bev_valid      [100,100] bool
        bev_fov_mask   [100,100] bool
    """

    UNOBSERVED = 255  # sentinel value in bev_semantics

    def __init__(
        self,
        data_root,
        split="train",
        img_size=(360, 640),
        val_count=200,
        seed=42,
    ):
        """
        Args:
            data_root: Path to 'Sample Dataset With Semantic Annotations'.
            split: "train" or "val".
            img_size: (H, W) to resize RGB images.
            val_count: Number of frames to hold out for validation.
            seed: Random seed for reproducible train/val split.
        """
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.img_size = img_size  # (H, W)

        img_dir = os.path.join(data_root, "pylon_camera_node", "0000")
        bev_dir = os.path.join(data_root, "bev_targets", "0000")

        # Match image ↔ BEV target by filename stem
        bev_files = {
            os.path.splitext(os.path.basename(f))[0]: f
            for f in glob.glob(os.path.join(bev_dir, "*.npz"))
        }
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

        self.pairs = []
        for img_path in img_files:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            if stem in bev_files:
                self.pairs.append((img_path, bev_files[stem]))

        # Deterministic random train/val split
        rng = np.random.RandomState(seed)
        indices = np.arange(len(self.pairs))
        rng.shuffle(indices)
        val_indices = set(indices[:val_count].tolist())

        if split == "val":
            self.pairs = [self.pairs[i] for i in sorted(val_indices)]
        else:
            self.pairs = [
                self.pairs[i]
                for i in range(len(self.pairs))
                if i not in val_indices
            ]

        # Camera intrinsics adjusted for img_size
        self.K = self._build_intrinsics()

        # Camera extrinsics (vehicle/lidar → camera)
        self.T_veh_cam = np.array([
            [0.0, -1.0, 0.0, 0.2],
            [0.0, 0.0, -1.0, 0.275],
            [1.0, 0.0, 0.0, -0.355],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        print(f"BEVDataset [{split}]: {len(self.pairs)} samples, img_size={img_size}")

    # ------------------------------------------------------------------

    def _build_intrinsics(self):
        """
        Builds 4×4 intrinsic matrix, scaled for the network input resolution.
        Original intrinsics are for 1440×1080 camera.
        Images on disk are 1280×720 (center-crop-H + V-scale).
        We further resize to self.img_size.
        """
        # Step 1: 1440×1080 → 1280×720
        crop_x = (1440 - 1280) / 2  # 80
        sy_disk = 720.0 / 1080.0     # 2/3

        fx_disk = 1037.350
        fy_disk = 1124.614 * sy_disk
        cx_disk = 708.762 - crop_x
        cy_disk = 549.905 * sy_disk

        # Step 2: 1280×720 → img_size (H, W)
        target_h, target_w = self.img_size
        sx = target_w / 1280.0
        sy = target_h / 720.0

        K = np.eye(4, dtype=np.float32)
        K[0, 0] = fx_disk * sx
        K[1, 1] = fy_disk * sy
        K[0, 2] = cx_disk * sx
        K[1, 2] = cy_disk * sy
        return K

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, bev_path = self.pairs[idx]

        # --- RGB image ---
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, H, W]

        # --- BEV targets ---
        bev = np.load(bev_path)
        bev_sem = torch.from_numpy(bev["bev_semantics"].astype(np.int64))  # [100, 100]
        bev_height = torch.from_numpy(bev["bev_height"])                    # [100, 100]
        bev_slope = torch.from_numpy(bev["bev_slope"])                      # [100, 100]
        bev_valid = torch.from_numpy(bev["bev_valid"])                      # [100, 100]
        bev_fov = torch.from_numpy(bev["bev_fov_mask"])                     # [100, 100]

        # Supervision mask: valid AND inside FOV
        sup_mask = bev_valid & bev_fov

        return {
            "image": img_tensor,             # [3, H, W]
            "intrinsics": torch.from_numpy(self.K),  # [4, 4]
            "extrinsics": torch.from_numpy(self.T_veh_cam),  # [4, 4]
            "bev_semantics": bev_sem,        # [100, 100] int64 (255 = unobserved)
            "bev_height": bev_height,        # [100, 100] float32
            "bev_slope": bev_slope,          # [100, 100] float32
            "bev_valid": sup_mask,           # [100, 100] bool
            "bev_fov_mask": bev_fov,         # [100, 100] bool
        }


if __name__ == "__main__":
    HOME = os.path.expanduser("~")
    root = os.path.join(HOME, "Downloads", "Sample Dataset With Semantic Annotations")

    for split in ("train", "val"):
        ds = BEVDataset(root, split=split)
        if len(ds) > 0:
            sample = ds[0]
            print(f"\n{split} sample shapes:")
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape} {v.dtype}")
