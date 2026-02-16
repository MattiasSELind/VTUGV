import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def make_camera_frustum_grid(H, W, z_range, num_bins, K, device='cpu'):
    """
    Generate a 3D grid of points aligned with camera rays (Frustum).
    
    Args:
        H, W: Image height, width
        z_range: (min, max) depth
        num_bins: Number of depth bins
        K: (3, 3) Intrinsic matrix (assumed same for batch if not batched input)
        device: torch device
        
    Returns:
        grid_coords: (1, 3, H, W, D) 3D coordinates in Camera Frame
        z_vals: (1, 1, 1, 1, D) Depth values
    """
    # 1. Create pixel grid
    # Downsample H, W could be passed in, assuming H, W are feature map sizes
    u = torch.linspace(0, W - 1, W, device=device)
    v = torch.linspace(0, H - 1, H, device=device)
    vv, uu = torch.meshgrid(v, u, indexing='ij') # (H, W)
    
    # 2. Logarithmic depth bins
    z_min, z_max = z_range
    z_log_min = np.log(z_min)
    z_log_max = np.log(z_max)
    z_vals = torch.exp(torch.linspace(z_log_min, z_log_max, num_bins, device=device)) # (D)
    
    # 3. Valid homogenous pixels
    # (3, H*W)
    ones = torch.ones_like(uu)
    pixels = torch.stack([uu, vv, ones], dim=0).reshape(3, -1)
    
    # 4. Unproject to 3D Rays
    # K_inv @ pixels -> Rays
    # We need K to be on device
    K = K.to(device)
    K_inv = torch.inverse(K)
    rays = K_inv @ pixels # (3, H*W)
    
    # 5. Expand by Depth
    # rays: (3, HW), z: (D)
    # points = rays * z
    # (3, HW, 1) * (1, 1, D) -> (3, HW, D)
    rays = rays.unsqueeze(2)
    z_vals_expanded = z_vals.view(1, 1, -1)
    
    points_3d = rays * z_vals_expanded # (3, HW, D)
    
    # Reshape to (3, H, W, D)
    points_3d = points_3d.reshape(3, H, W, num_bins)
    
    # Add batch dim: (1, 3, H, W, D)
    return points_3d.unsqueeze(0), z_vals.view(1, 1, 1, 1, -1)


def volume_render(density, color, z_vals):
    """
    Volume rendering (NeRF-style) along rays.
    
    Args:
        density: (B, 1, H, W, D) Positive density values
        color: (B, C, H, W, D) Color values
        z_vals: (B, 1, 1, 1, D) Depth values along the ray
        
    Returns:
        depth_map: (B, 1, H, W) Expected depth
        rgb_map: (B, C, H, W) Expected color
        weights: (B, 1, H, W, D) Weights for each sample
    """
    # Delta Z (distance between samples)
    # We append a large value for the last segment
    dists = z_vals[..., 1:] - z_vals[..., :-1] # (..., D-1)
    
    # Concatenate a large distance (1e10) for the last sample
    last_dist = torch.tensor([1e10], device=z_vals.device).expand(dists[..., :1].shape)
    dists = torch.cat([dists, last_dist], dim=-1) # (..., D)
    
    # Expand dists to spatial dims if needed (it is 1,1,1,1,D currently)
    # density is (B, 1, H, W, D)
    
    # Alpha = 1 - exp(-sigma * delta)
    alpha = 1.0 - torch.exp(-density * dists)
    
    # Transmittance = cumprod(1 - alpha)
    # Weights = T * alpha
    # We filter exclusive cumprod: [1, (1-a0), (1-a0)(1-a1), ...]
    
    # Prepend 1.0
    # (B, 1, H, W, D+1)
    ones = torch.ones_like(alpha[..., :1])
    trans = torch.cat([ones, 1.0 - alpha + 1e-10], dim=-1)
    trans = torch.cumprod(trans, dim=-1)[..., :-1] # Drop the last one
    
    weights = alpha * trans # (B, 1, H, W, D)
    
    # Integrate Depth
    depth_map = torch.sum(weights * z_vals, dim=-1) # (B, 1, H, W)
    
    # Integrate Color
    rgb_map = torch.sum(weights * color, dim=-1) # (B, C, H, W)
    
    return depth_map, rgb_map, weights


class SpatialCrossAttention(nn.Module):
    """
    Projects 3D queries to 2D and samples features using bilinear interpolation.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # Optional: Learnable attention weights or fusion layer
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, queries_3d, feature_map, K, T_vehicle_to_cam):
        """
        Args:
            queries_3d: (B, 3, N) 3D coordinates in vehicle frame
            feature_map: (B, C, H, W) 2D image features
            K: (B, 3, 3) Camera intrinsics
            T_vehicle_to_cam: (B, 4, 4) Extrinsics
            
        Returns:
            sampled_features: (B, C, N)
            valid_mask: (B, 1, N)
        """
        B, C, H, W = feature_map.shape
        _, _, N = queries_3d.shape
        
        # 1. Transform Vehicle -> Camera
        # queries_3d is (B, 3, N)
        ones = torch.ones((B, 1, N), device=queries_3d.device)
        pts_homo = torch.cat([queries_3d, ones], dim=1) # (B, 4, N)
        
        # T_vehicle_to_cam: (B, 4, 4)
        pts_cam = T_vehicle_to_cam @ pts_homo
        pts_cam = pts_cam[:, :3, :] # (B, 3, N)
        
        # 2. Project Camera -> Image
        pts_img = K @ pts_cam
        z = pts_img[:, 2:3, :].clamp(min=1e-5)
        u = pts_img[:, 0:1, :] / z
        v = pts_img[:, 1:2, :] / z
        
        # 3. Normalize for grid_sample
        u_norm = 2.0 * u / (W - 1) - 1.0
        v_norm = 2.0 * v / (H - 1) - 1.0
        
        # Stack: (B, 1, N, 2)
        grid = torch.cat([u_norm, v_norm], dim=1).permute(0, 2, 1).unsqueeze(1)
        
        # 4. Sample
        sampled = F.grid_sample(feature_map, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled = sampled.squeeze(2) # (B, C, N)
        
        # 5. Mask
        in_front = (pts_cam[:, 2:3, :] > 0.1)
        in_bounds = (u_norm >= -1) & (u_norm <= 1) & (v_norm >= -1) & (v_norm <= 1)
        valid_mask = in_front & in_bounds
        
        return sampled, valid_mask


class QueryOccNet(nn.Module):
    def __init__(self, 
                 z_range=(2, 60), 
                 num_bins=64,
                 feature_dim=128,
                 img_size=(544, 1024)):
        super().__init__()
        
        self.z_range = z_range
        self.num_bins = num_bins
        self.feature_dim = feature_dim
        self.img_size = img_size # (H, W)
        
        # Downsample factor for feature map (e.g., 8 for ResNet18 layer 2)
        self.downsample = 8
        self.feat_h = int(img_size[0] / self.downsample)
        self.feat_w = int(img_size[1] / self.downsample)
        
        # 1. 3D Query Embedding (Learnable)
        # One feature vector per frustum voxel
        self.query_embed = nn.Parameter(
            torch.randn(1, feature_dim, self.feat_h, self.feat_w, num_bins) * 0.02
        )
        
        # 2. Image Backbone
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, stride=2), # H/2
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), # H/4
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, feature_dim, 3, padding=1, stride=2), # H/8
            nn.BatchNorm2d(feature_dim), nn.ReLU()
        )
        
        # 3. Spatial Cross Attention
        self.spatial_cross_attn = SpatialCrossAttention(feature_dim)
        
        # 4. Heads
        self.density_head = nn.Sequential(
            nn.Conv3d(feature_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, 1),
            nn.Softplus() # Ensure positive density
        )
        
        self.color_head = nn.Sequential(
            nn.Conv3d(feature_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 3, 1),
            nn.Sigmoid() # RGB [0,1]
        )

    def forward(self, images, poses, K, T_cam_to_veh):
        """
        Args:
            images: list of tensors [(B, 3, H, W)] for [prev, curr, next]
            poses: list of tensors [(B, 4, 4)] world poses for [prev, curr, next]
            K: (B, 3, 3) Intrinsics
            T_cam_to_veh: (B, 4, 4) Static transform
            
        Returns:
            rendered_depth, rendered_rgb
        """
        img_curr = images[1]
        pose_curr = poses[1]
        B = img_curr.shape[0]
        
        # 1. Generate Frustum Grid in Camera Frame
        # (1, 3, H, W, D)
        grid_cam, z_vals = make_camera_frustum_grid(
            self.feat_h, self.feat_w, self.z_range, self.num_bins, K[0], device=img_curr.device
        )
        grid_cam = grid_cam.expand(B, -1, -1, -1, -1)
        z_vals = z_vals.expand(B, -1, -1, -1, -1)
        
        # Flatten for transforming: (B, 3, N)
        B, _, H, W, D = grid_cam.shape
        grid_cam_flat = grid_cam.reshape(B, 3, -1)
        
        # Transform Grid to Current Vehicle Frame
        # grid_veh = T_c2v @ grid_cam
        # T_cam_to_veh is (B, 4, 4)
        ones = torch.ones((B, 1, H*W*D), device=img_curr.device)
        grid_cam_homo = torch.cat([grid_cam_flat, ones], dim=1)
        grid_veh_flat = (T_cam_to_veh @ grid_cam_homo)[:, :3, :] # (B, 3, N)
        
        # 2. Accumulate Features from all frames (Temporal Fusion)
        accum_feats = 0
        total_weight = 0
        
        for i in range(len(images)):
            img = images[i] # (B, 3, H, W)
            pose_world = poses[i] # (B, 4, 4)
            
            # Extract features
            feat_map = self.encoder(img) # (B, C, Hf, Wf)
            
            # Compute Relative Transform: Curr Vehicle -> Target Camera
            # We want to project our grid points (in Curr Vehicle) to Target Camera
            # T_target_cam_from_curr_veh = T_target_cam_from_world @ T_world_from_curr_veh
            # T_target_cam_from_world = inv(T_target_veh_from_world @ T_target_cam_from_target_veh)
            # Actually we typically have T_world_from_vehicle.
            # T_c2v is static.
            
            # Target Camera Pose in World: T_world_from_tgt_cam = T_world_from_tgt_veh @ T_tgt_veh_from_tgt_cam
            # T_veh_to_cam == T_c2v.inverse()
            T_veh_to_cam = torch.inverse(T_cam_to_veh)
            
            # Grid points are in Curr Vehicle.
            # Transform: Curr Veh -> World -> Tgt Veh -> Tgt Cam
            
            # T_curr_veh_to_world = pose_curr
            # T_world_to_tgt_veh = inv(pose_world)
            
            T_world_to_tgt_veh = torch.inverse(pose_world)
            
            # Full chain: Curr Veh -> Tgt Cam
            # T = T_veh_to_cam @ T_world_to_tgt_veh @ T_curr_veh_to_world
            T_curr_veh_to_tgt_cam = T_veh_to_cam @ T_world_to_tgt_veh @ pose_curr
            
            # However, my SpatialCrossAttention expects T_vehicle_to_cam
            # It takes points in "Vehicle" (which is the input frame) and transforms to "Camera".
            # So passing T_curr_veh_to_tgt_cam directly as T_vehicle_to_cam works!
            
            sampled, mask = self.spatial_cross_attn(
                grid_veh_flat, feat_map, K, T_curr_veh_to_tgt_cam
            )
            
            accum_feats = accum_feats + sampled * mask
            total_weight = total_weight + mask
            
        # Average features
        avg_feats = accum_feats / (total_weight + 1e-6)
        
        # 3. Add Queries
        # (B, C, H, W, D)
        avg_feats = avg_feats.reshape(B, self.feature_dim, H, W, D)
        query_embed = self.query_embed.expand(B, -1, -1, -1, -1)
        
        volumetric_feats = avg_feats + query_embed
        
        # 4. Heads
        density = self.density_head(volumetric_feats) # (B, 1, H, W, D)
        color = self.color_head(volumetric_feats)     # (B, 3, H, W, D)
        
        # 5. Volume Render
        depth_pred, rgb_pred, weights = volume_render(density, color, z_vals)
        
        # 6. Upsample to original image size
        # depth_pred: (B, 1, Hf, Wf) -> (B, 1, H, W)
        # rgb_pred: (B, 3, Hf, Wf) -> (B, 3, H, W)
        if (depth_pred.shape[-2:] != self.img_size):
            depth_pred = F.interpolate(depth_pred, size=self.img_size, mode='bilinear', align_corners=False)
            rgb_pred = F.interpolate(rgb_pred, size=self.img_size, mode='bilinear', align_corners=False)
            
        return depth_pred, rgb_pred
