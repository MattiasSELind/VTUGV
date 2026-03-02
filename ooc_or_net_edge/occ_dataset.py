"""
queryocc_dataset.py
PyTorch Dataset for self-supervised QueryOcc training.
Loads time-synchronized 2D RGB images, DINOv2 features, semantic masks, and Odometry poses.
"""

import os
import glob


class OffRoadOccDataset(Dataset):
    def __init__(self, data_root, split="train", img_size=(256, 512), 
                 max_time_diff_ns=50_000_000, # 50 milliseconds tolerance
                 seq_len=2): # Number of continuous frames per sample
        """
        Args:
            data_root: Root directory of the preprocessed dataset.
            split: "train" or "val" (can be used to split the temporal sequence).
            img_size: Resize loaded RGB images and semantic masks to this (H, W).
            max_time_diff_ns: Maximum allowed nanosecond difference between an image timestamp and the nearest odometry pose.
            seq_len: Temporal sequence chunk size for handling recurrent temporal fusion (e.g., ConvGRU).
        """
        self.data_root = data_root
        self.img_size = img_size
        self.max_time_diff = max_time_diff_ns
        self.seq_len = seq_len
        
        # Paths
        self.images_dir = os.path.join(data_root, "images")
        self.features_dir = os.path.join(data_root, "features")
        self.semantics_dir = os.path.join(data_root, "semantics", "front_left")
        self.lidar_depth_dir = os.path.join(data_root, "bev_targets", "front_left") # NPZ LiDAR depth mapping
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
        """Hardcoded front_left camera intrinsics."""
        K_mat = np.eye(4, dtype=np.float32)
        K_mat[0, 0] = 1037.350 # FX
        K_mat[1, 1] = 1124.614 # FY
        K_mat[0, 2] = 708.762  # CX
        K_mat[1, 2] = 549.905  # CY
                
        return K_mat

    def _load_extrinsics(self):
        """Hardcoded Lidar/Vehicle -> Camera extrinsics."""
        # T_lidar_to_camera (Vehicle -> Camera)
        # Derived from transformslid2left_cam.yaml (q=[0.5, -0.5, 0.5, -0.5], t=[0.355, -0.2, -0.275])
        T_mat = np.array([
            [ 0.0, -1.0,  0.0,  0.2   ],
            [ 0.0,  0.0, -1.0,  0.275 ],
            [ 1.0,  0.0,  0.0, -0.355 ],
            [ 0.0,  0.0,  0.0,  1.0   ]
        ], dtype=np.float32)

        return T_mat

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
        for every valid synchronized timestep for the single front camera.
        """
        samples = []
        
        # 1. Collect all available feature timestamps
        cam = "front_left"
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

        for i in range(len(times_arr)):
            t_ref = times_arr[i]
            bn_ref = bns_arr[i]
            
            world_pose = self._get_nearest_pose(t_ref)
            if world_pose is None:
                continue
                
            sample_data = {
                "timestamp": t_ref,
                "world_pose": world_pose,
                "basename": bn_ref,
                "cameras": {}
            }
            
            # Check for either .jpg or .png
            img_path_jpg = os.path.join(self.images_dir, cam, f"{bn_ref}.jpg")
            img_path_png = os.path.join(self.images_dir, cam, f"{bn_ref}.png")
            
            img_path = img_path_jpg if os.path.exists(img_path_jpg) else img_path_png
            feat_path = os.path.join(self.features_dir, cam, f"{bn_ref}.npz")
            sem_path = os.path.join(self.semantics_dir, cam, f"{bn_ref}.npy")
            
            # Check for depth map
            depth_path = os.path.join(self.lidar_depth_dir, f"{bn_ref}.npz")
            
            if not os.path.exists(img_path) or not os.path.exists(feat_path):
                print(f"DEBUG: Dropped {bn_ref} because missing img or feat for {cam}.")
                continue
                
            sample_data["cameras"][cam] = {
                "image": img_path,
                "features": feat_path,
                "semantics": sem_path if os.path.exists(sem_path) else None,
                "depth": depth_path if os.path.exists(depth_path) else None
            }
            
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
        final_gt_lidar_depth = None
        final_gt_lidar_valid = None
        
        for step in range(self.seq_len):
            sample = self.samples[start_idx + step]
            
            # The base vehicle world pose at this exact time step
            T_world_veh = sample["world_pose"]
            seq_world_poses.append(torch.from_numpy(T_world_veh).float())
            seq_basenames.append(sample["basename"])
            
            # Generate optional target BEV output only for the LAST timestep
            # because BPTT accumulates gradient exclusively backward from the final sequence state
            is_final_step = (step == self.seq_len - 1)
        
            cam = "front_left"
            cam_data = sample["cameras"][cam]
            
            # 1. Load RGB Image
            img = Image.open(cam_data["image"]).convert("RGB")
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1) # [3, H, W]
            seq_imgs.append(img_tensor)
            
            # 2. Load DINOv2 Features
            # Stored as shape (H_patch, W_patch, feat_dim) half precision float16
            feat_data = np.load(cam_data["features"])["features"] 
            feat_tensor = torch.from_numpy(feat_data.astype(np.float32)).permute(2, 0, 1) # [feat_dim, H_patch, W_patch]
            seq_feats.append(feat_tensor)
            
            # 3. Load Semantics (if available)
            if cam_data["semantics"] is not None:
                sem = np.load(cam_data["semantics"]) # [H_orig, W_orig] uint8
                sem_img = Image.fromarray(sem).resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
                sem_tensor = torch.from_numpy(np.array(sem_img)).long() # [H, W]
            else:
                sem_tensor = torch.zeros(self.img_size, dtype=torch.long)
            seq_sems.append(sem_tensor)
            
            # 4. Vehicle to Camera Extrinsics
            # The network operates in the Vehicle frame, so we just need T_vehicle_to_camera
            T_veh_cam = self.T_veh_cam
            seq_poses.append(torch.from_numpy(T_veh_cam).float())
            
            # 5. Intrinsics
            K_mat = self.K
            seq_intrinsics.append(torch.from_numpy(K_mat).float())
            
            # 6. Load Downsampled Depth for Explicit depth distillation (Method 2)
            # and Full Res Depth for building Target Pseudo-BEV Ground Truth
            if cam_data["depth"] is not None:
                # Load the .npz file generated by generate_bev_targets.py
                depth_data = np.load(cam_data["depth"])
                
                # 1. Grab full res depth for projecting Semantics/Features to BEV
                depth_np = depth_data["depth_map_full"] # [256, 512] float32 array in meters
                
                if is_final_step:
                    # 2. Grab downsampled depth target and valid mask for d_loss training
                    final_gt_lidar_depth = torch.from_numpy(depth_data["gt_lidar_depth_2d"]).unsqueeze(0) # [1, fH, fW]
                    final_gt_lidar_valid = torch.from_numpy(depth_data["gt_lidar_valid"]).unsqueeze(0)    # [1, fH, fW]
                
                # Ensure full res maps matches exactly what we scaled our semantics to
                # Even though shape might match already, this guards against accidental size changes
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
                
                # Apply strictly 39-degree Front Camera Horizontal FOV Mask 
                # FOV is 39 degrees total -> 19.5 degrees left and 19.5 degrees right of center (Z-axis).
                # Note: In our BEV grid format, X is forward/backward, Y is left/right.
                # Assuming vehicle is at the origin (X=0, Y=0), so in grid coords:
                # Vehicle origin is at: grid_x = (0 - (-25)) / 1 = 25, grid_y = (0 - (-25)) / 1 = 25
                veh_grid_x = 25.0
                veh_grid_y = 25.0
                
                # Calculate angle in radians from the camera center line (forward X axis)
                # Angle = arctan(delta_y / delta_x)
                dx = idx_x.float() - veh_grid_x
                dy = idx_y.float() - veh_grid_y
                
                # Angles in degrees
                angles_deg = torch.atan2(torch.abs(dy), torch.max(dx, torch.tensor(1e-6))) * (180.0 / np.pi)
                
                # FOV mask: Angle must be <= 19.5 degrees and it must be strictly in front of the car (dx > 0)
                fov_mask = (angles_deg <= 19.5) & (dx > 0.0)
                
                # Combine bounding box mask with FOV mask
                valid_mask = in_bounds & fov_mask
                
                idx_x = idx_x[valid_mask]
                idx_y = idx_y[valid_mask]
                sem_valid = sem_valid[valid_mask]
                
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
                    
                    # --- Compute Occupancy & Traversability Cost ---
                    # Based on the 13 ClipSeg classes + 1 Padding
                    # 0: dirt road, 1: grass, 2: tree, 3: bush, 4: rock, 5: mud, 6: water
                    # 7: sky, 8: vehicle, 9: person, 10: building, 11: fence, 12: terrain, 13: padding
                    
                    # 1. Traversability Cost [0.0 = safe, 1.0 = lethal]
                    final_gt_bev_cost = torch.zeros_like(final_gt_bev_sem, dtype=torch.float32)
                    class_cost_mapping = {
                        0: 0.0,   # dirt road
                        1: 0.2,   # grass (safe but bumpy)
                        2: 1.0,   # tree (lethal)
                        3: 0.8,   # bush (high friction, avoid if possible)
                        4: 1.0,   # rock (lethal)
                        5: 0.6,   # mud (high slip risk)
                        6: 1.0,   # water (lethal sink)
                        7: 0.0,   # sky (irrelevant to ground planner, cost 0)
                        8: 1.0,   # vehicle (lethal obstacle)
                        9: 1.0,   # person (lethal obstacle)
                        10: 1.0,  # building (lethal structure)
                        11: 1.0,  # fence (lethal barrier)
                        12: 0.1,  # terrain (generic flat ground)
                        13: 0.5   # padding/unknown (moderate caution)
                    }
                    for c_idx, c_cost in class_cost_mapping.items():
                        final_gt_bev_cost[final_gt_bev_sem == c_idx] = c_cost
                        
                    # 2. Occupancy (0=Free Space, 1=Obstacle)
                    # We define an "obstacle" as anything with a cost >= 0.8
                    final_gt_bev_occ = (final_gt_bev_cost >= 0.8).float()
                
        # Stack all lists to tensors [seq_len, ...]
        output_batch = {
            "images": torch.stack(seq_imgs),                 # [S, 3, H, W]
            "features": torch.stack(seq_feats),              # [S, D, Hp, Wp]
            "semantics": torch.stack(seq_sems),              # [S, H, W]
            "poses": torch.stack(seq_poses),                 # [S, 4, 4] -> T_veh_camera
            "intrinsics": torch.stack(seq_intrinsics),       # [S, 4, 4]
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
            
            # Extract final timestep 2D depth for multi-task LiDAR distillation
            # The last element in _temps lists corresponds to S-1
            output_batch["gt_lidar_depth_2d"] = final_gt_lidar_depth # [1, fH, fW]
            output_batch["gt_lidar_valid"] = final_gt_lidar_valid    # [1, fH, fW]
            
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
