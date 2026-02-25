"""
off_road_occ_net.py
Core PyTorch Model for Self-Supervised QueryOcc.
Lifts 2D RGB images into a 3D Semantic Occupancy Volume using Spatial Cross-Attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetBackbone(nn.Module):
    """
    Lightweight 2D backbone to extract features from input RGB images.
    Returns features at 1/8th or 1/16th resolution depending on configuration.
    """
    def __init__(self, out_channels=256):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        # Use layers up to layer3 (1/16 resolution) or layer2 (1/8 resolution)
        self.backbone = nn.Sequential(*list(resnet.children())[:6]) # layer2
        self.conv_out = nn.Conv2d(128, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x: [B, C, H, W]
        features = self.backbone(x)
        features = self.conv_out(features)
        return features


class SpatialCrossAttention(nn.Module):
    """
    Lifts 2D features into 3D by projecting 3D voxel queries into the 2D image plane 
    and sampling the corresponding features using bilinear interpolation.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        # Optional mapping layers can be added here (e.g., query/key/value projections)

    def forward(self, queries_3d, features_2d, intrinsics, extrinsics):
        """
        Args:
            queries_3d: [B, 3, Z, Y, X] 3D coordinates in the vehicle frame.
            features_2d: [B, num_cams, C, H, W] Extracted 2D features from backbone.
            intrinsics: [B, num_cams, 4, 4] Camera intrinsics.
            extrinsics: [B, num_cams, 4, 4] Camera extrinsics (Vehicle to Camera).
        Returns:
            lifted_features: [B, C, Z, Y, X]
        """
        B, num_cams, C, H, W = features_2d.shape
        _, _, Z, Y, X = queries_3d.shape
        
        # Output tensor
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
            # Matrix mult: [B, 4, 4] @ [B, 4, N] -> [B, 4, N]
            P_cam = torch.bmm(T_veh_cam, coords_h)
            P_uv = torch.bmm(K, P_cam)
            
            # Dehomogenize
            z = P_uv[:, 2, :]
            u = P_uv[:, 0, :] / (z + 1e-6)
            v = P_uv[:, 1, :] / (z + 1e-6)
            
            # Valid mask (front of camera only)
            valid = (z > 0.1) & (z < 100.0)
            
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
                lifted_volume[b].view(C, N)[:, v_mask] += sampled_feat[b][:, v_mask]
                hit_counts[b].view(1, N)[:, v_mask] += 1.0

        # Average where hit_counts > 0
        valid_hits = hit_counts > 0
        lifted_volume[valid_hits.repeat(1, C, 1, 1, 1)] /= hit_counts[valid_hits].repeat(C).view(-1)
        
        return lifted_volume

class VoxelGrid(nn.Module):
    """
    Generates the physical 3D coordinates for the query volume.
    """
    def __init__(self, x_bounds=[-50, 50], y_bounds=[-50, 50], z_bounds=[-2, 6], resolution=0.5):
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


class Decoder3D(nn.Module):
    """
    3D CNN to process the lifted voxel features, smooth artifacts,
    and output the final semantic distribution and feature logits.
    """
    def __init__(self, in_channels, num_semantic_classes, dinov2_feat_dim):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(128)
        self.conv2 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        
        # Semantic Head (Predicts [0.0 - 1.0] probability for each class)
        self.semantic_head = nn.Conv3d(64, num_semantic_classes, kernel_size=1)
        
        # DINOv2 Feature Reconstruction Head
        self.feature_head = nn.Conv3d(64, dinov2_feat_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, C, Z, Y, X]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        semantic_logits = self.semantic_head(x) # [B, num_classes, Z, Y, X]
        feature_logits = self.feature_head(x)   # [B, dinov2_dim, Z, Y, X]
        
        return semantic_logits, feature_logits


class OffRoadOccNet(nn.Module):
    """
    Complete Off-Road QueryOcc Model.
    """
    def __init__(self, num_semantic_classes=14, dinov2_feat_dim=384, embed_dim=256):
        super().__init__()
        
        self.backbone = ResNetBackbone(out_channels=embed_dim)
        
        # 100x100m grid, 8m tall, 1m resolution = 100 x 100 x 8 volume
        self.voxel_grid = VoxelGrid(x_bounds=[-50, 50], y_bounds=[-50, 50], z_bounds=[-2, 6], resolution=0.5)
        
        self.cross_attention = SpatialCrossAttention(embed_dim=embed_dim)
        
        self.decoder = Decoder3D(
            in_channels=embed_dim, 
            num_semantic_classes=num_semantic_classes, 
            dinov2_feat_dim=dinov2_feat_dim
        )
        
    def forward(self, images, intrinsics, extrinsics):
        """
        Args:
            images: [B, num_cams, 3, H, W]
            intrinsics: [B, num_cams, 4, 4]
            extrinsics: [B, num_cams, 4, 4] Transforms Vehicle -> Camera
        Returns:
            semantic_volume: [B, num_classes, Z, Y, X]
            feature_volume: [B, dinov2_dim, Z, Y, X]
        """
        B, num_cams, C, H, W = images.shape
        
        # 1. Image Backbone
        images_flat = images.view(B * num_cams, C, H, W)
        features_flat = self.backbone(images_flat) # [B*num_cams, embed_dim, H', W']
        
        _, embed_dim, fH, fW = features_flat.shape
        features = features_flat.view(B, num_cams, embed_dim, fH, fW)
        
        # 2. Generate 3D Queries
        queries_3d = self.voxel_grid(batch_size=B) # [B, 3, Z, Y, X]
        
        # 3. Spatial Cross Attention (Lifting)
        # Note: We pass the feature shape (fH, fW) to the attention module implicitly
        #       but intrinsics need to be scaled if the backbone downsizes the image!
        #       For resnet layer2, stride is 8.
        scale_factor = 1.0 / 8.0
        scaled_intrinsics = intrinsics.clone()
        scaled_intrinsics[:, :, 0, :] *= scale_factor # fx, cx
        scaled_intrinsics[:, :, 1, :] *= scale_factor # fy, cy
        
        lifted_volume = self.cross_attention(queries_3d, features, scaled_intrinsics, extrinsics)

        # 4. 3D Decoder Refinement
        semantic_volume, feature_volume = self.decoder(lifted_volume)
        
        return semantic_volume, feature_volume

if __name__ == "__main__":
    # Quick shape sanity check
    B, num_cams = 2, 3
    H, W = 256, 512
    
    # Mock inputs
    images = torch.randn(B, num_cams, 3, H, W)
    
    # Mock Intrinsics K
    K = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, num_cams, 1, 1)
    K[:, :, 0, 0] = 477.0 # fx
    K[:, :, 1, 1] = 477.0 # fy
    K[:, :, 0, 2] = 256.0 # cx
    K[:, :, 1, 2] = 128.0 # cy
    
    # Mock Extrinsics (Vehicle -> Camera)
    T_v_c = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, num_cams, 1, 1)
    
    model = OffRoadOccNet(num_semantic_classes=14, dinov2_feat_dim=384, embed_dim=256)
    
    print("Testing OffRoadOccNet forward pass...")
    semantic_vol, feature_vol = model(images, K, T_v_c)
    
    print(f"Semantic Volume Shape: {semantic_vol.shape}")
    print(f"Feature Volume Shape: {feature_vol.shape}")
    print("Test passed successfully!")
