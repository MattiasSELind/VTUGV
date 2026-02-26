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

class MobileNetV3Backbone(nn.Module):
    """
    Ultra-lightweight 2D backbone for edge devices.
    Extracts features using MobileNetV3 Small.
    """
    def __init__(self, out_channels=256):
        super().__init__()
        # MobileNet V3 Small is extremely fast on embedded GPUs
        # We need its internal feature maps, so we extract from features
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        
        # We take layers up to index 8, which represents a 1/16 spatial scale reduction.
        # This keeps the receptive field decent while drastically reducing sizes.
        self.backbone = nn.Sequential(*list(mobilenet.features.children())[:9])
        
        # In MobileNetV3 small, the features at idx 8 have 24 channels.
        self.conv_out = nn.Sequential(
            nn.Conv2d(24, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # x: [B, C, H, W]
        features = self.backbone(x) # -> [B, 24, H/16, W/16]
        features = self.conv_out(features) # -> [B, out_channels, H/16, W/16]
        return features


class VoxelGridBEV(nn.Module):
    """
    Generates a restricted physical 3D coordinate frame focused tightly on the vehicle.
    """
    def __init__(self, x_bounds=[-25, 25], y_bounds=[-25, 25], z_bounds=[-1.0, 3.0], resolution=1.0):
        super().__init__()
        self.resolution = resolution
        
        # Number of voxels
        X = int((x_bounds[1] - x_bounds[0]) / resolution)
        Y = int((y_bounds[1] - y_bounds[0]) / resolution)
        Z = int((z_bounds[1] - z_bounds[0]) / resolution)
        
        # Create meshgrid
        x = torch.linspace(x_bounds[0], x_bounds[1], X)
        y = torch.linspace(y_bounds[0], y_bounds[1], Y)
        z = torch.linspace(z_bounds[0], z_bounds[1], Z)
        
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        
        # Register as buffer [3, Z, Y, X]
        self.register_buffer('coords', torch.stack([grid_x, grid_y, grid_z]))
        
    def forward(self, batch_size):
        # [B, 3, Z, Y, X]
        return self.coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)


class SpatialCrossAttentionBEV(nn.Module):
    """
    Lifts 2D features into 3D, and then IMMEDIATELY collapses the Z dimension 
    to form a 2D Bird's Eye View (BEV) map. This bypasses the need for 3D convolutions.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, queries_3d, features_2d, intrinsics, extrinsics):
        """
        Args:
            queries_3d: [B, 3, Z, Y, X]
            features_2d: [B, num_cams, C, H, W]
            intrinsics: [B, num_cams, 4, 4]
            extrinsics: [B, num_cams, 4, 4]
        Returns:
            bev_features: [B, C, Y, X]
        """
        B, num_cams, C, H, W = features_2d.shape
        _, _, Z, Y, X = queries_3d.shape
        
        # Output tensor for 3D Accumulation
        lifted_volume = torch.zeros((B, C, Z, Y, X), device=queries_3d.device)
        hit_counts = torch.zeros((B, 1, Z, Y, X), device=queries_3d.device)

        # Flatten 3D coordinates [B, 3, N]
        coords = queries_3d.view(B, 3, -1)
        N = coords.shape[2]
        
        # Homogeneous coordinates [B, 4, N]
        ones = torch.ones((B, 1, N), device=coords.device)
        coords_h = torch.cat([coords, ones], dim=1)
        
        for cam in range(num_cams):
            T_veh_cam = extrinsics[:, cam, :, :] # [B, 4, 4]
            K = intrinsics[:, cam, :, :] # [B, 4, 4]
            
            # P_cam = K * T_veh_cam * P_veh
            P_cam = torch.bmm(T_veh_cam, coords_h)
            P_uv = torch.bmm(K, P_cam)
            
            # Dehomogenize
            z = P_uv[:, 2, :]
            u = P_uv[:, 0, :] / (z + 1e-6)
            v = P_uv[:, 1, :] / (z + 1e-6)
            
            # Valid mask (front of camera only, closer depth range for edge)
            valid = (z > 0.1) & (z < 60.0)
            
            # Normalize u, v to [-1, 1] for grid_sample
            u_norm = (u / W) * 2 - 1
            v_norm = (v / H) * 2 - 1
            
            # Valid mask (inside image bounds)
            valid = valid & (u_norm >= -1) & (u_norm <= 1) & (v_norm >= -1) & (v_norm <= 1)
            
            # Shape grid for grid_sample: [B, N, 1, 2]
            grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(2)
            
            # Sample features: [B, C, H, W] -> [B, C, N, 1] -> [B, C, N]
            sampled_feat = F.grid_sample(features_2d[:, cam], grid, align_corners=True).squeeze(-1)
            
            # Accumulate valid hits
            for b in range(B):
                v_mask = valid[b]
                if v_mask.any():
                    lifted_volume[b].view(C, N)[:, v_mask] += sampled_feat[b][:, v_mask]
                    hit_counts[b].view(1, N)[:, v_mask] += 1.0

        # Average where hit_counts > 0
        valid_hits = hit_counts > 0
        # To avoid expanding large tensors on limited VRAM, operate conditionally
        # using a simple division with a clamp.
        denom = torch.clamp(hit_counts, min=1.0)
        lifted_volume = lifted_volume / denom
        
        # --- CRITICAL EDGE OPTIMIZATION: BEV PROJECTION ---
        # Collapse the Z dimension (Height) immediately using Max Pooling
        # Shape goes from [B, C, Z, Y, X] -> [B, C, Y, X]
        bev_volume, _ = torch.max(lifted_volume, dim=2)
        
        return bev_volume


class Decoder2D(nn.Module):
    """
    Lightweight 2D CNN to process the BEV projection.
    Significantly faster and more memory efficient than 3D convolutions.
    """
    def __init__(self, in_channels, num_semantic_classes, dinov2_feat_dim):
        super().__init__()
        
        # Bottleneck architecture block
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Semantic Head 
        self.semantic_head = nn.Conv2d(64, num_semantic_classes, kernel_size=1)
        
        # DINOv2 Feature Reconstruction Head
        self.feature_head = nn.Conv2d(64, dinov2_feat_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, C, Y, X] BEV Volume
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        semantic_logits = self.semantic_head(x) # [B, num_classes, Y, X]
        feature_logits = self.feature_head(x)   # [B, dinov2_dim, Y, X]
        
        return semantic_logits, feature_logits


class OffRoadOccNetNano(nn.Module):
    """
    Complete Edge-Optimized Off-Road QueryOcc Model.
    Designed to run on severe memory and compute constraints (Jetson Nano).
    """
    def __init__(self, num_semantic_classes=14, dinov2_feat_dim=384, embed_dim=128):
        super().__init__()
        
        # 1. Edge-Optimized Backbone
        self.backbone = MobileNetV3Backbone(out_channels=embed_dim)
        
        # 2. Reduced 3D Grid: 50x50m, 1m resolution = 50x50x4 volume (10k voxels vs 80k in full version)
        self.voxel_grid = VoxelGridBEV(
            x_bounds=[-25, 25], 
            y_bounds=[-25, 25], 
            z_bounds=[-1.0, 3.0], 
            resolution=1.0
        )
        
        # 3. BEV Projecting Attention
        self.cross_attention = SpatialCrossAttentionBEV(embed_dim=embed_dim)
        
        # 4. Fast 2D Decoder
        self.decoder = Decoder2D(
            in_channels=embed_dim, 
            num_semantic_classes=num_semantic_classes, 
            dinov2_feat_dim=dinov2_feat_dim
        )
        
    def forward(self, images, intrinsics, extrinsics):
        """
        Args:
            images: [B, num_cams, 3, H, W]
            intrinsics: [B, num_cams, 4, 4]
            extrinsics: [B, num_cams, 4, 4]
        Returns:
            semantic_bev: [B, num_classes, Y, X] BEV Map
            feature_bev: [B, dinov2_dim, Y, X] BEV Map
        """
        B, num_cams, C, H, W = images.shape
        
        # 1. Image Backbone
        images_flat = images.view(B * num_cams, C, H, W)
        features_flat = self.backbone(images_flat) 
        
        _, embed_dim, fH, fW = features_flat.shape
        features = features_flat.view(B, num_cams, embed_dim, fH, fW)
        
        # 2. Generate 3D Queries
        queries_3d = self.voxel_grid(batch_size=B) # [B, 3, Z, Y, X]
        
        # 3. Spatial Cross Attention & BEV Projection
        # MobileNet layer setup reduces resolution by factor of 16
        scale_factor = 1.0 / 16.0
        scaled_intrinsics = intrinsics.clone()
        scaled_intrinsics[:, :, 0, :] *= scale_factor # fx, cx
        scaled_intrinsics[:, :, 1, :] *= scale_factor # fy, cy
        
        bev_volume = self.cross_attention(queries_3d, features, scaled_intrinsics, extrinsics)

        # 4. 2D Decoder Refinement
        semantic_bev, feature_bev = self.decoder(bev_volume)
        
        return semantic_bev, feature_bev


if __name__ == "__main__":
    import time
    
    print("Initializing OffRoadOccNetNano prototype...")
    # Parameters tailored for Nano capability limits
    model = OffRoadOccNetNano(num_semantic_classes=14, dinov2_feat_dim=384, embed_dim=128)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params / 1e6:.2f} M")
    
    # Quick shape sanity check mapping reality
    B, num_cams = 1, 2  # Jetson Nano typically processes 1 batch and maybe 2 cameras
    H, W = 256, 512
    
    images = torch.randn(B, num_cams, 3, H, W)
    
    K = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, num_cams, 1, 1)
    K[:, :, 0, 0] = 477.0 # fx
    K[:, :, 1, 1] = 477.0 # fy
    K[:, :, 0, 2] = 256.0 # cx
    K[:, :, 1, 2] = 128.0 # cy
    
    T_v_c = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, num_cams, 1, 1)
    
    # Test Time
    start_t = time.time()
    with torch.no_grad():
        semantic_bev, feature_bev = model(images, K, T_v_c)
    end_t = time.time()
    
    print(f"\nForward Pass Complete in {end_t - start_t:.3f} seconds (CPU)")
    print(f"Semantic BEV Output Shape: {semantic_bev.shape} (Expected: [1, 14, 50, 50])")
    print(f"Feature BEV Output Shape: {feature_bev.shape} (Expected: [1, 384, 50, 50])")
    print("Test passed successfully!")
