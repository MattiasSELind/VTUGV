"""
off_road_occ_net_nano.py
Optimized Edge Prototype for Jetson Nano.
Lifts 2D RGB image features into a 3D grid, but immediately pools into a 2D BEV map
to allow for fast, memory-efficient 2D decoder processing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EfficientNetBackbone(nn.Module):
    """
    Ultra-lightweight 2D backbone for edge devices.
    Extracts features using EfficientNet-B0.
    """
    def __init__(self, out_channels=256):
        super().__init__()
        # EfficientNet-B0 is fast and has a better accuracy-speed tradeoff
        effnet = models.efficientnet_b0(pretrained=True)
        
        # We take layers up to index 6, which represents a 1/16 spatial scale reduction.
        self.backbone = nn.Sequential(*list(effnet.features.children())[:6])
        
        # In EfficientNet-B0, the features after index 5 have 112 channels.
        self.conv_out = nn.Sequential(
            nn.Conv2d(112, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # x: [B, C, H, W]
        features = self.backbone(x) # -> [B, 112, H/16, W/16]
        features = self.conv_out(features) # -> [B, out_channels, H/16, W/16]
        return features



class DepthHead(nn.Module):
    """
    Continuous depth regression head to explicitly supervise the 2D backbone.
    Outputs a single channel representing absolute depth in meters.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
    def forward(self, x):
        # x: [B, C, H_feat, W_feat]
        # output: [B, 1, H_feat, W_feat]
        return torch.relu(self.conv(x)) # Depth must be positive


class SpatialAlignment(nn.Module):
    """
    Given the vehicle pose at t-1 and t, computes the 2D affine transformation
    required to align the previous BEV feature map to the current spatial frame.
    Assumes rotation primarily around the Z axis (yaw) and 2D translation (X,Y).
    """
    def __init__(self, x_bounds=[-25, 25], y_bounds=[-25, 25], resolution=1.0):
        super().__init__()
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.res = resolution
        
    def forward(self, prev_bev, prev_pose, curr_pose):
        """
        prev_bev: [B, C, Y, X] BEV feature map at t-1
        prev_pose: [B, 4, 4] T_world_veh at t-1
        curr_pose: [B, 4, 4] T_world_veh at t
        """
        if prev_bev is None:
            return None
            
        B, C, Y, X = prev_bev.shape
        device = prev_bev.device
        
        # Calculate relative transformation: T_curr_prev = (T_world_curr)^-1 * T_world_prev
        curr_pose_inv = torch.inverse(curr_pose)
        T_rel = torch.bmm(curr_pose_inv, prev_pose) # [B, 4, 4]
        
        # Extract 2D translation (x, y)
        dx = T_rel[:, 0, 3] # [B]
        dy = T_rel[:, 1, 3] # [B]
        
        # Extract 2D rotation (yaw angle)
        # Assuming rotation matrix primarily around Z. R_00 = cos(yaw), R_10 = sin(yaw)
        cos_theta = T_rel[:, 0, 0]
        sin_theta = T_rel[:, 1, 0]
        
        # Convert translation from metric space to normalized [-1, 1] grid space 
        # width = (x_max - x_min), height = (y_max - y_min)
        width_m = self.x_bounds[1] - self.x_bounds[0]
        height_m = self.y_bounds[1] - self.y_bounds[0]
        
        # Normalizing logic: map real-world delta to pixel grid offset
        # Note: direction of affine transform might need tuning depending on coordinate system conventions
        tx = -2.0 * dx / width_m
        ty = -2.0 * dy / height_m
        
        # Build affine transformation matrix [B, 2, 3]
        theta = torch.zeros((B, 2, 3), device=device)
        theta[:, 0, 0] = cos_theta
        theta[:, 0, 1] = -sin_theta # Affine rotation sign flips
        theta[:, 0, 2] = tx
        
        theta[:, 1, 0] = sin_theta
        theta[:, 1, 1] = cos_theta
        theta[:, 1, 2] = ty
        
        # Warp the previous BEV state
        grid = F.affine_grid(theta, prev_bev.size(), align_corners=False)
        aligned_bev = F.grid_sample(prev_bev, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        return aligned_bev


class ConvGRU(nn.Module):
    """
    Lightweight 2D Convolutional GRU for fusing temporal BEV features.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        padding = kernel_size // 2
        
        # Update and Reset Gates
        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2 * hidden_dim,
                                    kernel_size=kernel_size,
                                    padding=padding)
        
        # New State Proposition
        self.conv_state = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=hidden_dim,
                                    kernel_size=kernel_size,
                                    padding=padding)
                                    
    def forward(self, x, h_prev=None):
        """
        x: [B, C, Y, X] current BEV observation
        h_prev: [B, C, Y, X] previous hidden state (warped)
        """
        if h_prev is None:
            # First frame, state is just the current observation
            return x
            
        # Concatenate on channel dim
        combined = torch.cat([x, h_prev], dim=1) # [B, 2C, Y, X]
        
        # Calculate gates
        gates = self.conv_gates(combined) # [B, 2*hidden, Y, X]
        update_gate, reset_gate = torch.split(gates, self.hidden_dim, dim=1)
        
        z_t = torch.sigmoid(update_gate)
        r_t = torch.sigmoid(reset_gate)
        
        # Proposition state
        combined_reset = torch.cat([x, r_t * h_prev], dim=1)
        h_prop = torch.tanh(self.conv_state(combined_reset))
        
        # Final output
        h_t = (1 - z_t) * h_prev + z_t * h_prop
        
        return h_t


class LSSViewTransformer(nn.Module):
    """
    Deterministic Lift-Splat-Shoot Module for Edge Deployment.
    Lifts 2D pixels to 3D using explicit continuous depth predictions,
    and spatters (scatters) them directly into the 2D BEV map.
    Memory efficiency: O(H*W) instead of O(D*H*W).
    """
    def __init__(self, x_bounds=[-25, 25], y_bounds=[-25, 25], z_bounds=[-1.0, 3.0], resolution=1.0):
        super().__init__()
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.res = resolution
        
        self.X = int((x_bounds[1] - x_bounds[0]) / resolution)
        self.Y = int((y_bounds[1] - y_bounds[0]) / resolution)
        self.Z = int((z_bounds[1] - z_bounds[0]) / resolution)
        
    def forward(self, features, depths, intrinsics, extrinsics):
        """
        Args:
            features: [B, C, fH, fW] 2D Backbone Embeddings
            depths: [B, 1, fH, fW] Absolute depth in meters
            intrinsics: [B, 4, 4] Intrinsics at the fH/fW scale
            extrinsics: [B, 4, 4] T_veh_camera
        Returns:
            bev_volume: [B, C, Y, X] 
        """
        B, C, fH, fW = features.shape
        device = features.device
        
        # 1. Create a flattened pixel grid [fH, fW] -> [fH*fW, 2]
        y, x = torch.meshgrid(torch.arange(fH, device=device), torch.arange(fW, device=device), indexing='ij')
        # Homogeneous pixel coordinates [3, fH*fW]
        pix_coords = torch.stack([x.float(), y.float(), torch.ones_like(x).float()], dim=0).view(3, -1)
        
        bev_volume = torch.zeros((B, C, self.Y, self.X), device=device)
        
        for b in range(B):
            feat_c = features[b] # [C, fH, fW]
            depth_c = depths[b]  # [1, fH, fW]
            K_c = intrinsics[b]  # [4, 4]
            T_vc = extrinsics[b] # [4, 4]
            
            # --- LIFT (2D -> 3D Camera) ---
            K_inv_3x3 = torch.inverse(K_c[:3, :3])
            
            # Multiply pixel rays by inverse intrisics and absolute depth
            # P_cam = K^-1 * uv_homog * Z
            rays = torch.matmul(K_inv_3x3, pix_coords) # [3, fH*fW]
            depth_flat = depth_c.view(1, -1) # [1, fH*fW]
            P_cam = rays * depth_flat # [3, fH*fW]
            
            # Filter valid depths
            valid_depth = (depth_flat > 0.5) & (depth_flat < 50.0) # [1, N]
            P_cam_valid = P_cam[:, valid_depth.squeeze(0)] # [3, N_valid]
            feat_valid = feat_c.view(C, -1)[:, valid_depth.squeeze(0)] # [C, N_valid]
            
            if P_cam_valid.shape[1] == 0:
                continue
                
            # Homogenize Camera Points
            ones = torch.ones((1, P_cam_valid.shape[1]), device=device)
            P_cam_homog = torch.cat([P_cam_valid, ones], dim=0) # [4, N_valid]
            
            # Transform to Vehicle Frame
            P_veh = torch.matmul(T_vc, P_cam_homog) # [4, N_valid]
            
            # Filter points within the physical BEV grid bounds
            x_v = P_veh[0, :]
            y_v = P_veh[1, :]
            z_v = P_veh[2, :]
            
            valid_bounds = (x_v >= self.x_bounds[0]) & (x_v < self.x_bounds[1]) & \
                           (y_v >= self.y_bounds[0]) & (y_v < self.y_bounds[1]) & \
                           (z_v >= self.z_bounds[0]) & (z_v < self.z_bounds[1])
                           
            P_veh_valid = P_veh[:3, valid_bounds] # [3, N_final]
            feat_final = feat_valid[:, valid_bounds] # [C, N_final]
            
            if P_veh_valid.shape[1] == 0:
                continue
                
            # Aggregate all valid frustum points for this batch element
            all_coords = P_veh_valid # [3, N_total]
            all_feats = feat_final # [C, N_total]
            
            # --- SPLAT (3D -> 2D Grid Accumulation) ---
            # Digitize to grid indices
            idx_x = torch.floor((all_coords[0, :] - self.x_bounds[0]) / self.res).long()
            idx_y = torch.floor((all_coords[1, :] - self.y_bounds[0]) / self.res).long()
            
            # Scatter features into the [C, Y, X] grid
            # Compute a flat index [Y * X] to use with scatter_add_
            flat_indices = (idx_y * self.X + idx_x).view(1, -1).expand(C, -1) # [C, N_total]
            
            bev_flat = torch.zeros((C, self.Y * self.X), device=device)
            bev_flat.scatter_add_(dim=1, index=flat_indices, src=all_feats)
            
            # Hit count for averaging
            hit_flat = torch.zeros((1, self.Y * self.X), device=device)
            ones_src = torch.ones((1, all_feats.shape[1]), device=device)
            hit_flat.scatter_add_(dim=1, index=flat_indices[0:1, :], src=ones_src)
            
            # Average colliding features
            bev_flat = bev_flat / torch.clamp(hit_flat, min=1.0)
            
            bev_volume[b] = bev_flat.view(C, self.Y, self.X)
            
        return bev_volume


class TraversabilityHead(nn.Module):
    """
    Lightweight CNN head to output Occupancy, Semantics, Features, and Traversability Cost.
    FlashOcc style: 2D processing directly on the BEV map.
    """
    def __init__(self, in_channels, num_semantic_classes, dinov2_feat_dim):
        super().__init__()
        
        # Shared bottleneck to reduce channels and expand receptive field
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Branch 1: Semantics
        self.semantic_head = nn.Conv2d(64, num_semantic_classes, kernel_size=1)
        
        # Branch 2: DINOv2 Features
        self.feature_head = nn.Conv2d(64, dinov2_feat_dim, kernel_size=1)
        
        # Branch 3: Occupancy (1 channel logits - BCEWithLogits)
        self.occupancy_head = nn.Conv2d(64, 1, kernel_size=1)
        
        # Branch 4: Traversability Cost (1 channel [0, 1] - Sigmoid applied inside/outside)
        # Using Sigmoid so output is bounded for continuous MSE/L1 loss
        self.cost_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.shared_conv(x)
        sem = self.semantic_head(features)
        feat = self.feature_head(features)
        occ = self.occupancy_head(features)
        cost = self.cost_head(features)
        return occ, sem, feat, cost


class OffRoadOccNetEdge(nn.Module):
    """
    Complete Edge-Optimized Off-Road QueryOcc Model.
    Designed to run on severe memory and compute constraints (Jetson Nano/Orin).
    """
    def __init__(self, num_semantic_classes=14, dinov2_feat_dim=384, embed_dim=128):
        super().__init__()
        
        # 1. Edge-Optimized Backbone
        self.backbone = EfficientNetBackbone(out_channels=embed_dim)
        
        # 2. Lift-Splat View Transformer
        self.lss = LSSViewTransformer(
            x_bounds=[-25, 25], 
            y_bounds=[-25, 25], 
            z_bounds=[-1.0, 3.0], 
            resolution=1.0
        )
        
        # 3. Temporal Fusion Module
        self.aligner = SpatialAlignment(x_bounds=[-25, 25], y_bounds=[-25, 25], resolution=1.0)
        self.temporal_fusion = ConvGRU(input_dim=embed_dim, hidden_dim=embed_dim, kernel_size=3)
        
        # 4. Traversability Head
        self.decoder = TraversabilityHead(
            in_channels=embed_dim, 
            num_semantic_classes=num_semantic_classes, 
            dinov2_feat_dim=dinov2_feat_dim
        )
        
        # 5. Auxiliary 2D Depth Head
        self.depth_head = DepthHead(in_channels=embed_dim)
        
    def forward(self, images, intrinsics, extrinsics, prev_bev=None, prev_pose=None, curr_pose=None):
        """
        Args:
            images: [B, 3, H, W]
            intrinsics: [B, 4, 4]
            extrinsics: [B, 4, 4]
            prev_bev: [B, C, Y, X] Hidden state from previous timestamp
            prev_pose: [B, 4, 4] World pose at t-1
            curr_pose: [B, 4, 4] World pose at t
        Returns:
            occ_logits: [B, 1, Y, X] Occupancy grid map (unnormalized logits)
            semantic_bev: [B, num_classes, Y, X] BEV Map
            feature_bev: [B, dinov2_dim, Y, X] BEV Map
            cost_map: [B, 1, Y, X] Traversability Cost Map [0, 1]
            depths_2d: [B, 1, fH, fW] Auxiliary Depth Features
            curr_bev_state: [B, C, Y, X] Raw BEV feature map representing recurrent state t
        """
        B, C, H, W = images.shape
        
        # 1. Image Backbone
        features = self.backbone(images) 
        
        # Auxiliary Depth Head Prediction
        depths_2d = self.depth_head(features)
        
        # 2. Lift-Splat Forward Projection
        scale_factor = 1.0 / 16.0
        scaled_intrinsics = intrinsics.clone()
        scaled_intrinsics[:, 0, :] *= scale_factor # fx, cx
        scaled_intrinsics[:, 1, :] *= scale_factor # fy, cy
        
        bev_volume = self.lss(features, depths_2d, scaled_intrinsics, extrinsics)
        
        # 3. Temporal Alignment and Fusion
        aligned_prev_bev = None
        if prev_bev is not None and prev_pose is not None and curr_pose is not None:
            aligned_prev_bev = self.aligner(prev_bev, prev_pose, curr_pose)
            
        curr_bev_state = self.temporal_fusion(bev_volume, aligned_prev_bev)

        # 4. Traversability Head Refinement
        occ_logits, semantic_bev, feature_bev, cost_map = self.decoder(curr_bev_state)
        
        return occ_logits, semantic_bev, feature_bev, cost_map, depths_2d, curr_bev_state


if __name__ == "__main__":
    import time
    
    print("Initializing OffRoadOccNetEdge prototype...")
    # Parameters tailored for Nano capability limits
    model = OffRoadOccNetEdge(num_semantic_classes=14, dinov2_feat_dim=384, embed_dim=128)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params / 1e6:.2f} M")
    
    # Quick shape sanity check mapping reality
    B = 1  # Jetson Nano typically processes 1 batch
    H, W = 256, 512
    
    images = torch.randn(B, 3, H, W)
    
    K = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    K[:, 0, 0] = 477.0 # fx
    K[:, 1, 1] = 477.0 # fy
    K[:, 0, 2] = 256.0 # cx
    K[:, 1, 2] = 128.0 # cy
    
    T_v_c = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    
    # Dummy world poses for temporal processing
    T_w_prev = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    T_w_curr = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    # Simulate a 1 meter forward movement
    T_w_curr[:, 0, 3] = 1.0 
    
    # Test Time
    start_t = time.time()
    with torch.no_grad():
        occ_t0, sem_t0, feat_t0, cost_t0, d_t0, state_t0 = model(images, K, T_v_c)
        # Test recurrent pass
        occ_t1, sem_t1, feat_t1, cost_t1, d_t1, state_t1 = model(images, K, T_v_c, prev_bev=state_t0, prev_pose=T_w_prev, curr_pose=T_w_curr)
    end_t = time.time()
    
    print(f"\nForward Pass Complete in {end_t - start_t:.3f} seconds (CPU) [Two timesteps]")
    print(f"Occupancy Output Shape: {occ_t1.shape} (Expected: [1, 1, 50, 50])")
    print(f"Semantic BEV Output Shape: {sem_t1.shape} (Expected: [1, 14, 50, 50])")
    print(f"Feature BEV Output Shape: {feat_t1.shape} (Expected: [1, 384, 50, 50])")
    print(f"Cost Map Output Shape: {cost_t1.shape} (Expected: [1, 1, 50, 50])")
    print(f"Depth 2D Output Shape: {d_t1.shape} (Expected: [1, 1, 16, 32])")
    print(f"Temporal Internal State Output: {state_t1.shape}")
    print("Test passed successfully!")
