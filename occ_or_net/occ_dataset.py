"""
queryocc_dataset.py
PyTorch Dataset for self-supervised QueryOcc training.
Loads time-synchronized 2D RGB images, DINOv2 features, semantic masks, and Odometry poses.
"""

import os
import glob
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

def parse_yaml_transform(path):
    """Parses a simple YAML file into q and t dictionaries."""
    q = {}
    t = {}
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("q:"):
                current_section = "q"
            elif line.startswith("t:"):
                current_section = "t"
            elif ":" in line:
                key, val = line.split(":")
                key = key.strip()
                val = val.strip()
                if current_section == "q" and key in ['w', 'x', 'y', 'z']:
                    q[key] = float(val)
                elif current_section == "t" and key in ['x', 'y', 'z']:
                    t[key] = float(val)
    except Exception as e:
        print(f"Error parsing YAML {path}: {e}")
        return None, None
    return q, t

def quat_to_mat(q, t):
    """Convert quaternion (w,x,y,z) and translation (x,y,z) to 4x4 matrix."""
    w, x, y, z = q['w'], q['x'], q['y'], q['z']
    tx, ty, tz = t['x'], t['y'], t['z']
    
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    R_mat = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ])
    
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = [tx, ty, tz]
    return T

class OffRoadOccDataset(Dataset):
    def __init__(self, data_root, split="train", img_size=(256, 512), 
                 cameras=["front_left", "front_right", "rear_center"],
                 max_time_diff_ns=50_000_000): # 50 milliseconds tolerance
        """
        Args:
            data_root: Root directory of the preprocessed dataset.
            split: "train" or "val" (can be used to split the temporal sequence).
            img_size: Resize loaded RGB images and semantic masks to this (H, W).
            cameras: List of camera identifiers.
            max_time_diff_ns: Maximum allowed nanosecond difference between an image timestamp and the nearest odometry pose.
        """
        self.data_root = data_root
        self.img_size = img_size
        self.cameras = cameras
        self.max_time_diff = max_time_diff_ns
        
        # Paths
        self.images_dir = os.path.join(data_root, "images")
        self.features_dir = os.path.join(data_root, "features")
        self.semantics_dir = os.path.join(data_root, "semantics")
        self.odom_file = os.path.join(data_root, "odometry", "odometry_filtered_odom.csv")
        
        # Load actual camera intrinsics
        self.K = self._load_intrinsics()
        
        # Load actual camera extrinsics (Vehicle -> Camera)
        self.T_veh_cam = self._load_extrinsics()
        
        print("Loading odometry database...")
        self.odom_times, self.odom_poses = self._load_odometry()
        
        print("Synchronizing camera frames...")
        self.samples = self._build_samples()
        
        # Split logic (e.g., 80% train, 20% validation)
        split_idx = int(len(self.samples) * 0.8)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
            
        print(f"Dataset [{split}] initialized with {len(self.samples)} synchronized tuples.")

    def _load_intrinsics(self):
        """Loads camera intrinsic matrices from .txt files into a dictionary."""
        intrinsics = {}
        mapping = {
            "front_left": "left_cam_intrinsic.txt",
            "front_right": "right_cam_intrinsic.txt",
            "rear_center": "rear_cam_intrinsic.txt"
        }
        
        path_base = os.path.join(self.data_root, "camera_intrinsics", "camera_intrinsics")
        
        for cam in self.cameras:
            filename = mapping.get(cam)
            if not filename:
                continue
                
            path = os.path.join(path_base, filename)
            K_mat = np.eye(4, dtype=np.float32)
            
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        line = f.readline().strip()
                        parts = line.split()
                        if len(parts) >= 4:
                            fx, fy, cx, cy = map(float, parts[:4])
                            K_mat[0, 0] = fx
                            K_mat[1, 1] = fy
                            K_mat[0, 2] = cx
                            K_mat[1, 2] = cy
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"Warning: Intrinsic file not found: {path} (Using identity)")
                
            intrinsics[cam] = K_mat
        return intrinsics

    def _load_extrinsics(self):
        """Loads Camera -> Lidar/Vehicle extrinsics from .yaml files."""
        extrinsics = {}
        mapping = {
            "front_left": "transformslid2left_cam.yaml",
            "front_right": "transformslid2right_cam.yaml",
            "rear_center": "transformslid2rear.yaml"
        }
        
        path_base = os.path.join(self.data_root, "camera2lidar", "camera2lidar")
        
        for cam in self.cameras:
            filename = mapping.get(cam)
            if not filename:
                continue
                
            path = os.path.join(path_base, filename)
            T_mat = np.eye(4, dtype=np.float32)
            
            if os.path.exists(path):
                q, t = parse_yaml_transform(path)
                if q and t:
                    # YAML files are T_camera_to_lidar
                    T_c_l = quat_to_mat(q, t)
                    
                    # We need T_lidar_to_camera (Vehicle -> Camera)
                    # T_l_c = inv(T_c_l)
                    T_l_c = np.linalg.inv(T_c_l)
                    T_mat = T_l_c.astype(np.float32)
            else:
                print(f"Warning: Extrinsic file not found: {path} (Using identity)")
                
            extrinsics[cam] = T_mat
        return extrinsics

    def _load_odometry(self):
        """Loads all poses from the odometry CSV."""
        df = pd.read_csv(self.odom_file)
        times = (df['timestamp_sec'].values * 1e9 + df['timestamp_nsec'].values).astype(np.int64)
        
        poses = []
        for _, row in df.iterrows():
            pos = [row['pos_x'], row['pos_y'], row['pos_z']]
            quat = [row['orient_x'], row['orient_y'], row['orient_z'], row['orient_w']]
            
            T = np.eye(4)
            T[:3, 3] = pos
            T[:3, :3] = R.from_quat(quat).as_matrix()
            poses.append(T)
            
        # Ensure sorted by time
        sort_idx = np.argsort(times)
        return times[sort_idx], np.array(poses)[sort_idx]

    def _get_nearest_pose(self, query_ts):
        """Returns the 4x4 world pose for a given nanosecond timestamp."""
        idx = np.searchsorted(self.odom_times, query_ts)
        
        # Boundary checks
        if idx == 0:
            diff = abs(query_ts - self.odom_times[0])
            return self.odom_poses[0] if diff <= self.max_time_diff else None
        if idx == len(self.odom_times):
            diff = abs(query_ts - self.odom_times[-1])
            return self.odom_poses[-1] if diff <= self.max_time_diff else None
            
        # Check adjacent elements
        diff_before = abs(query_ts - self.odom_times[idx-1])
        diff_after = abs(query_ts - self.odom_times[idx])
        
        if diff_before < diff_after:
            min_diff = diff_before
            best_pose = self.odom_poses[idx-1]
        else:
            min_diff = diff_after
            best_pose = self.odom_poses[idx]
            
        if min_diff <= self.max_time_diff:
            return best_pose
        return None

    def _build_samples(self):
        """
        Builds a list of dictionaries containing available file paths and world poses
        for every valid synchronized timestep.
        """
        samples = []
        ref_cam = self.cameras[0]
        
        # Load the timestamps.csv for the reference camera
        ref_ts_file = os.path.join(self.images_dir, ref_cam, "timestamps.csv")
        if not os.path.exists(ref_ts_file):
            print(f"Warning: Timestamps file not found for {ref_cam} at {ref_ts_file}")
            return []
            
        df_ref = pd.read_csv(ref_ts_file)
        
        # Optional: load timestamps for other cameras to find exact matches
        # For simplicity in this example, we assume frames across cameras
        # have exactly matching 'frame_XXXXXX.png' prefixes.
        
        for _, row in df_ref.iterrows():
            ts = int(row['timestamp_sec'] * 1e9 + row['timestamp_nsec'])
            filename = row['filename']
            basename = os.path.splitext(filename)[0] # format: 'frame_XXXXXX'
            
            # Find closest odometry pose
            world_pose = self._get_nearest_pose(ts)
            
            # If no valid pose is found within the time threshold, skip this frame
            if world_pose is None:
                continue
                
            sample_data = {
                "timestamp": ts,
                "world_pose": world_pose,
                "basename": basename,
                "cameras": {}
            }
            
            # Check file existence across all cameras
            all_cams_valid = True
            for cam in self.cameras:
                img_path = os.path.join(self.images_dir, cam, f"{basename}.png")
                feat_path = os.path.join(self.features_dir, cam, f"{basename}.npz")
                sem_path = os.path.join(self.semantics_dir, cam, f"{basename}.npy")
                
                # We require at minimum the image and features
                if not os.path.exists(img_path) or not os.path.exists(feat_path):
                    all_cams_valid = False
                    break
                    
                sample_data["cameras"][cam] = {
                    "image": img_path,
                    "features": feat_path,
                    "semantics": sem_path if os.path.exists(sem_path) else None
                }
                
            if all_cams_valid:
                samples.append(sample_data)
                
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Prepare batched tensors across the N cameras
        # Shapes will be: [num_cams, Channels, H, W]
        imgs = []
        feats = []
        sems = []
        poses = []
        intrinsics = []
        
        # The base vehicle world pose at this exact time step
        T_world_veh = sample["world_pose"]
        
        for cam in self.cameras:
            cam_data = sample["cameras"][cam]
            
            # 1. Load RGB Image
            img = Image.open(cam_data["image"]).convert("RGB")
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1) # [3, H, W]
            imgs.append(img_tensor)
            
            # 2. Load DINOv2 Features
            # Stored as shape (H_patch, W_patch, feat_dim) half precision float16
            feat_data = np.load(cam_data["features"])["features"] 
            feat_tensor = torch.from_numpy(feat_data.astype(np.float32)).permute(2, 0, 1) # [feat_dim, H_patch, W_patch]
            feats.append(feat_tensor)
            
            # 3. Load Semantics (if available)
            if cam_data["semantics"] is not None:
                sem = np.load(cam_data["semantics"]) # [H_orig, W_orig] uint8
                sem_img = Image.fromarray(sem).resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
                sem_tensor = torch.from_numpy(np.array(sem_img)).long() # [H, W]
            else:
                sem_tensor = torch.zeros(self.img_size, dtype=torch.long)
            sems.append(sem_tensor)
            
            # 4. Global World Pose of the Camera
            # T_world_cam = T_world_veh * T_veh_cam
            T_veh_cam = self.T_veh_cam.get(cam, np.eye(4, dtype=np.float32))
            T_world_cam = T_world_veh @ T_veh_cam
            poses.append(torch.from_numpy(T_world_cam).float())
            
            # 5. Intrinsics
            K_mat = self.K.get(cam, np.eye(4, dtype=np.float32))
            intrinsics.append(torch.from_numpy(K_mat).float())
            
        # Stack all lists to tensors
        return {
            "images": torch.stack(imgs),       # [N, 3, H, W]
            "features": torch.stack(feats),    # [N, D, Hp, Wp]
            "semantics": torch.stack(sems),    # [N, H, W]
            "poses": torch.stack(poses),       # [N, 4, 4] -> T_world_camera
            "intrinsics": torch.stack(intrinsics), # [N, 4, 4]
            "basename": sample["basename"]
        }

if __name__ == "__main__":
    # Test the dataset instantiation
    HOME = os.path.expanduser("~")
    mock_data_root = os.path.join(HOME, "Downloads", "VTUGV_pointclouds_outdoors")
    
    # We create a dummy test object if paths exist, else skip silently
    if os.path.exists(mock_data_root):
        dataset = QueryOccDataset(mock_data_root, split="train")
        print(f"Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            batch = dataset[0]
            print(f"Batch shapes:")
            print(f"  Images: {batch['images'].shape}")
            print(f"  Features: {batch['features'].shape}")
            print(f"  Semantics: {batch['semantics'].shape}")
            print(f"  Poses: {batch['poses'].shape}")
