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
import torch.nn.functional as F
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
                 cameras=["front_left", "rear_center"],
                 max_time_diff_ns=50_000_000, # 50 milliseconds tolerance
                 seq_len=2): # Number of continuous frames per sample
        """
        Args:
            data_root: Root directory of the preprocessed dataset.
            split: "train" or "val" (can be used to split the temporal sequence).
            img_size: Resize loaded RGB images and semantic masks to this (H, W).
            cameras: List of camera identifiers.
            max_time_diff_ns: Maximum allowed nanosecond difference between an image timestamp and the nearest odometry pose.
            seq_len: Temporal sequence chunk size for handling recurrent temporal fusion (e.g., ConvGRU).
        """
        self.data_root = data_root
        self.img_size = img_size
        self.cameras = cameras
        self.max_time_diff = max_time_diff_ns
        self.seq_len = seq_len
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
        self.depth_dir = os.path.join(data_root, "depth_front") # Added depth path
        self.odom_file = os.path.join(data_root, "poses.csv")
        
        # Load actual camera intrinsics
        self.K = self._load_intrinsics()
        
        # Load actual camera extrinsics (Vehicle -> Camera)
        self.T_veh_cam = self._load_extrinsics()
        
        print("Loading odometry database...")
        self.odom_times, self.odom_poses = self._load_odometry()
        
        print("Synchronizing camera frames...")
        self.samples = self._build_samples()
        
        # We need sequence chunks, so effective length is len - seq_len + 1
        self.valid_indices = self._filter_sequences()
        
        # Split logic (e.g., 80% train, 20% validation)
        split_idx = int(len(self.valid_indices) * 0.8)
        if split == "train":
            self.valid_indices = self.valid_indices[:split_idx]
        else:
            self.valid_indices = self.valid_indices[split_idx:]
            
        print(f"Dataset [{split}] initialized with {len(self.valid_indices)} synchronized sequences of length {self.seq_len}.")

    def _load_intrinsics(self):
        """Loads camera intrinsic matrices from .txt files into a dictionary."""
        intrinsics = {}
        mapping = {
            "front_left": "left_cam_intrinsic.txt",
            "rear_center": "rear_cam_intrinsic.txt"
        }
        
        path_base = os.path.join(self.data_root, "camera_intrinsics") 
        
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
            "rear_center": "transformslid2rear.yaml"
        }
        
        path_base = os.path.join(self.data_root, "camera2lidar")
        
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
        times = df['timestamp_ns'].values.astype(np.int64)
        
        poses = []
        for _, row in df.iterrows():
            pos = [row['p_x'], row['p_y'], row['p_z']]
            quat = [row['q_x'], row['q_y'], row['q_z'], row['q_w']]
            
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
        import glob
        samples = []
        
        # We match frames across cameras by choosing a reference camera
        # and finding the nearest timestamps in the others.
        ref_cam = self.cameras[0]
        
        # 1. Collect all available feature timestamps per camera
        cam_timestamps = {}
        for cam in self.cameras:
            feat_dir = os.path.join(self.features_dir, cam)
            if not os.path.exists(feat_dir):
                print(f"DEBUG: Features directory not found mapping to {cam}: {feat_dir}")
                return []
            
            # The basename of .npz is the bag timestamp (e.g. 1713882442722758732.npz)
            feats = glob.glob(os.path.join(feat_dir, "*.npz"))
            ts_list = []
            for f in feats:
                bn = os.path.splitext(os.path.basename(f))[0]
                try:
                    ts_list.append((int(bn), bn))
                except ValueError:
                    pass
            
            ts_list.sort(key=lambda x: x[0])
            if not ts_list:
                print(f"No valid timestamps extracted for camera {cam}")
                return []
                
            times_arr = np.array([x[0] for x in ts_list], dtype=np.int64)
            bns_arr = [x[1] for x in ts_list]
            cam_timestamps[cam] = (times_arr, bns_arr)

        ref_times, ref_bns = cam_timestamps[ref_cam]
        
        for i in range(len(ref_times)):
            t_ref = ref_times[i]
            bn_ref = ref_bns[i]
            
            world_pose = self._get_nearest_pose(t_ref)
            if world_pose is None:
                # print(f"DEBUG: Dropped {bn_ref} because no Odom pose within {self.max_time_diff}ns.")
                continue
                
            sample_data = {
                "timestamp": t_ref,
                "world_pose": world_pose,
                "basename": bn_ref,
                "cameras": {}
            }
            
            all_cams_valid = True
            
            for cam in self.cameras:
                if cam == ref_cam:
                    closest_bn = bn_ref
                else:
                    times, bns = cam_timestamps[cam]
                    idx = np.searchsorted(times, t_ref)
                    
                    best_idx = idx
                    min_diff = float('inf')
                    
                    if idx < len(times):
                        min_diff = abs(times[idx] - t_ref)
                    if idx > 0 and abs(times[idx-1] - t_ref) < min_diff:
                        best_idx = idx - 1
                        min_diff = abs(times[idx-1] - t_ref)
                    # 50 ms max synchronization error
                    if min_diff > 50_000_000:
                        print(f"DEBUG: Dropped {bn_ref} because cam {cam} sync diff is {min_diff} > 50ms.")
                        all_cams_valid = False
                        break
                        
                    closest_bn = bns[best_idx]
                
                # Check for either .jpg or .png
                img_path_jpg = os.path.join(self.images_dir, cam, f"{closest_bn}.jpg")
                img_path_png = os.path.join(self.images_dir, cam, f"{closest_bn}.png")
                
                img_path = img_path_jpg if os.path.exists(img_path_jpg) else img_path_png
                feat_path = os.path.join(self.features_dir, cam, f"{closest_bn}.npz")
                sem_path = os.path.join(self.semantics_dir, cam, f"{closest_bn}.npy")
                
                if not os.path.exists(img_path) or not os.path.exists(feat_path):
                    print(f"DEBUG: Dropped {bn_ref} because missing img or feat for {cam}.")
                    all_cams_valid = False
                    break
                    
                # Check for depth map if this is the front camera
                depth_path = os.path.join(self.depth_dir, f"{closest_bn}.npy") if cam == "front_left" else None
                    
                sample_data["cameras"][cam] = {
                    "image": img_path,
                    "features": feat_path,
                    "semantics": sem_path if os.path.exists(sem_path) else None,
                    "depth": depth_path if depth_path and os.path.exists(depth_path) else None
                }
                
            if all_cams_valid:
                samples.append(sample_data)
                
                
        return samples

    def _filter_sequences(self):
        """
        Finds valid continuous sequence chunks. We assume samples are chronologically ordered.
        A sequence break occurs if consecutive samples are more than 1 second apart.
        """
        valid_indices = []
        for i in range(len(self.samples) - self.seq_len + 1):
            is_valid = True
            for j in range(1, self.seq_len):
                t_curr = self.samples[i + j]["timestamp"]
                t_prev = self.samples[i + j - 1]["timestamp"]
                # 1 billion nanoseconds = 1 second
                if t_curr - t_prev > 1_000_000_000:
                    is_valid = False
                    break
            
            if is_valid:
                valid_indices.append(i)
                
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        # Initialize sequence containers
        seq_imgs = []
        seq_feats = []
        seq_sems = []
        seq_poses = []
        seq_intrinsics = []
        seq_basenames = []
        seq_world_poses = []
        
        # Ground truth structures for the FINAL element of the sequence
        final_gt_bev_sem = None
        final_gt_bev_valid = None
        final_gt_bev_feat = None
        final_gt_bev_occ = None
        final_gt_bev_cost = None
        
        for step in range(self.seq_len):
            sample = self.samples[start_idx + step]
            
            # Prepare batched tensors across the N cameras for THIS timestamp
            # Shapes will be: [num_cams, Channels, H, W]
            imgs = []
            feats = []
            sems = []
            poses = []
            intrinsics = []
            
            # The base vehicle world pose at this exact time step
            T_world_veh = sample["world_pose"]
            seq_world_poses.append(torch.from_numpy(T_world_veh).float())
            seq_basenames.append(sample["basename"])
            
            # Generate optional target BEV output only for the LAST timestep
            # because BPTT accumulates gradient exclusively backward from the final sequence state
            is_final_step = (step == self.seq_len - 1)
        
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
            
            # 4. Vehicle to Camera Extrinsics
            # The network operates in the Vehicle frame, so we just need T_vehicle_to_camera
            T_veh_cam = self.T_veh_cam.get(cam, np.eye(4, dtype=np.float32))
            poses.append(torch.from_numpy(T_veh_cam).float())
            
            # 5. Intrinsics
            K_mat = self.K.get(cam, np.eye(4, dtype=np.float32))
            intrinsics.append(torch.from_numpy(K_mat).float())
            
            # 6. Build Target Pseudo-BEV Ground Truth (Front Camera Only)
            # We still need depth here uniquely to generate the dense gt_bev pointcloud, 
            # but we won't return the raw 2D depth back down to the training objective.
            
            if cam == "front_left" and cam_data["depth"] is not None:
                depth_np = np.load(cam_data["depth"]) # Should be [H_orig, W_orig] depth map in meters
                # Ensure it matches exactly what we scaled our semantics to
                depth_im = Image.fromarray(depth_np).resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
                depth = torch.from_numpy(np.array(depth_im)).float()
                
                # Assign 2D depth for pointcloud projection
                depth_valid = (depth > 0.5) & (depth < 50.0)
                
                # Project pixels to 3D using intrinsics
                H, W = self.img_size
                y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                
                fx, fy = K_mat[0, 0], K_mat[1, 1]
                cx, cy = K_mat[0, 2], K_mat[1, 2]
                
                # Camera Frame 3D Coordinates
                Z_c = depth
                X_c = (x.float() - cx) * Z_c / fx
                Y_c = (y.float() - cy) * Z_c / fy
                
                # Flatten valid points
                valid_depth = (Z_c > 0.5) & (Z_c < 50.0) # Filter out bad depth
                pts_cam = torch.stack([X_c[valid_depth], Y_c[valid_depth], Z_c[valid_depth]], dim=1) # [N, 3]
                sem_valid = sem_tensor[valid_depth] # [N]
                
                # Upsample DINOv2 Features to image size
                feat_upsampled = F.interpolate(feat_tensor.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False).squeeze(0) # [D, H, W]
                feat_valid = feat_upsampled[:, valid_depth].T # [N, D]
                
                # Transform to Vehicle Frame
                T_cam_veh = np.linalg.inv(T_veh_cam) # Because T_veh_cam goes from Veh -> Cam
                R_cv = torch.from_numpy(T_cam_veh[:3, :3]).float()
                t_cv = torch.from_numpy(T_cam_veh[:3, 3]).float()
                
                pts_veh = (R_cv @ pts_cam.T).T + t_cv # [N, 3]
                
                # Digitize into BEV Grid
                # Matching Jetson Nano Prototype bounds [-25, 25] at 1m resolution (50x50)
                # X points forward/back, Y points left/right in vehicle frame roughly
                x_bounds = [-25, 25]
                y_bounds = [-25, 25]
                res = 1.0
                
                idx_x = torch.floor((pts_veh[:, 0] - x_bounds[0]) / res).long()
                idx_y = torch.floor((pts_veh[:, 1] - y_bounds[0]) / res).long()
                
                # Filter points inside bounding box
                in_bounds = (idx_x >= 0) & (idx_x < 50) & (idx_y >= 0) & (idx_y < 50)
                
                idx_x = idx_x[in_bounds]
                idx_y = idx_y[in_bounds]
                sem_valid = sem_valid[in_bounds]
                
                # We only need the full GT structures mapped on the final timestep
                if is_final_step:
                    # Create the target grids
                    final_gt_bev_sem = torch.zeros((50, 50), dtype=torch.long)
                    final_gt_bev_valid = torch.zeros((50, 50), dtype=torch.bool)
                    D = feat_tensor.shape[0]
                    final_gt_bev_feat = torch.zeros((D, 50, 50), dtype=torch.float32)
                    
                    # Scatter the semantic ids into the grid. 
                    # (For simplicity, last written value wins if multiple points fall in voxel)
                    final_gt_bev_sem[idx_y, idx_x] = sem_valid
                    final_gt_bev_valid[idx_y, idx_x] = True
                    final_gt_bev_feat[:, idx_y, idx_x] = feat_valid.T
                    
                    # Compute Occupancy (0=Free, 1=Occupied)
                    # We assume class 0 is free space or background, >0 is occupied
                    final_gt_bev_occ = (final_gt_bev_sem > 0).float()
                    
                    # Compute Traversability Cost mapping
                    # (Dummy typical values: road=0.0, grass=0.3, mud=0.5, solid objects=1.0)
                    # User needs to adjust based on their specific 14 ClipSeg classes
                    final_gt_bev_cost = torch.zeros_like(final_gt_bev_sem, dtype=torch.float32)
                    class_cost_mapping = {
                        0: 0.0, 1: 0.0, 2: 0.2, 3: 0.5, 4: 1.0, 5: 1.0,
                        6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0,
                        12: 1.0, 13: 1.0, 14: 1.0
                    }
                    for c_idx, c_cost in class_cost_mapping.items():
                        final_gt_bev_cost[final_gt_bev_sem == c_idx] = c_cost
                
            seq_imgs.append(torch.stack(imgs))
            seq_feats.append(torch.stack(feats))
            seq_sems.append(torch.stack(sems))
            seq_poses.append(torch.stack(poses))
            seq_intrinsics.append(torch.stack(intrinsics))
            
        # Stack all lists to tensors [seq_len, ...]
        output_batch = {
            "images": torch.stack(seq_imgs),                 # [S, N, 3, H, W]
            "features": torch.stack(seq_feats),              # [S, N, D, Hp, Wp]
            "semantics": torch.stack(seq_sems),              # [S, N, H, W]
            "poses": torch.stack(seq_poses),                 # [S, N, 4, 4] -> T_veh_camera
            "intrinsics": torch.stack(seq_intrinsics),       # [S, N, 4, 4]
            "basename": seq_basenames,                       # List [S]
            "world_pose": torch.stack(seq_world_poses)       # [S, 4, 4] -> T_world_veh
        }
        
        # Add BEV targets for the final timestep
        if final_gt_bev_sem is not None:
            output_batch["gt_bev_semantic"] = final_gt_bev_sem # [50, 50]
            output_batch["gt_bev_valid"] = final_gt_bev_valid  # [50, 50]
            output_batch["gt_bev_feat"] = final_gt_bev_feat    # [D, 50, 50]
            output_batch["gt_bev_occ"] = final_gt_bev_occ      # [50, 50]
            output_batch["gt_bev_cost"] = final_gt_bev_cost    # [50, 50]
            
        return output_batch

if __name__ == "__main__":
    # Test the dataset instantiation
    HOME = os.path.expanduser("~")
    mock_data_root = os.path.join(HOME, "Downloads", "VTUGV_pointclouds_outdoors")
    
    # We create a dummy test object if paths exist, else skip silently
    if os.path.exists(mock_data_root):
        dataset = OffRoadOccDataset(mock_data_root, split="train", seq_len=2)
        print(f"Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            batch = dataset[0]
            print(f"Batch shapes:")
            print(f"  Images: {batch['images'].shape}")
            print(f"  Features: {batch['features'].shape}")
            print(f"  Poses: {batch['poses'].shape}")
            print(f"  World Poses: {batch['world_pose'].shape}")
