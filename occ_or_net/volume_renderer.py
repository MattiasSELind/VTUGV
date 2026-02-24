"""
volume_renderer.py
Implements Ray Generation and Volume Rendering (Alpha Compositing)
to project the 3D semantic/feature occupancy grid back into a 2D novel view.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VolumeRenderer(nn.Module):
    def __init__(self, x_bounds=[-50, 50], y_bounds=[-50, 50], z_bounds=[-2, 6],
                 resolution=1.0, num_samples=64):
        """
        Args:
            bounds: The physical limits of the 3D voxel grid (meters)
            resolution: The physical size of each voxel (meters)
            num_samples: Number of points to sample along each camera ray
        """
        super().__init__()
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.resolution = resolution
        self.num_samples = num_samples

    def get_rays(self, H, W, K, T_v_c):
        """
        Generates 3D rays corresponding to each pixel in an image.
        Args:
            H, W: Target image height and width
            K: [B, 4, 4] Intrinsics of target camera
            T_v_c: [B, 4, 4] Extrinsics (Vehicle -> Target Camera)
        Returns:
            ro: [B, H*W, 3] Ray origins in Vehicle frame
            rd: [B, H*W, 3] Ray directions in Vehicle frame
        """
        B = K.shape[0]
        
        # 1. Create 2D pixel grid
        y, x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=K.device),
            torch.arange(W, dtype=torch.float32, device=K.device),
            indexing='ij'
        )
        # [3, H*W]
        pixel_coords = torch.stack([x.flatten(), y.flatten(), torch.ones_like(x.flatten())], dim=0)
        pixel_coords = pixel_coords.unsqueeze(0).repeat(B, 1, 1) # [B, 3, H*W]
        
        # 2. Invert intrinsics to get ray directions in Camera frame
        # K is 4x4, we just need the 3x3 rotational/focal part
        K_inv = torch.inverse(K[:, :3, :3]) # [B, 3, 3]
        
        # [B, 3, 3] @ [B, 3, H*W] -> [B, 3, H*W]
        rd_cam = torch.bmm(K_inv, pixel_coords)
        
        # Normalize in camera frame
        rd_cam = rd_cam / torch.norm(rd_cam, dim=1, keepdim=True)
        
        # 3. Transform to Vehicle frame
        # T_c_v = inverse(T_v_c)
        T_c_v = torch.inverse(T_v_c)
        R_c_v = T_c_v[:, :3, :3] # [B, 3, 3]
        t_c_v = T_c_v[:, :3, 3].unsqueeze(2) # [B, 3, 1]
        
        # Directions: Rotate only
        rd_veh = torch.bmm(R_c_v, rd_cam) # [B, 3, H*W]
        rd_veh = rd_veh.transpose(1, 2) # [B, H*W, 3]
        
        # Origins: Same for all pixels, equal to translation
        ro_veh = t_c_v.transpose(1, 2).repeat(1, H*W, 1) # [B, H*W, 3]
        
        return ro_veh, rd_veh

    def sample_along_rays(self, ro, rd, near=1.0, far=60.0):
        """
        Stratified sampling of points along the rays.
        """
        B, num_rays, _ = ro.shape
        
        # Linearly spaced depths
        t_vals = torch.linspace(near, far, self.num_samples, device=ro.device)
        t_vals = t_vals.view(1, 1, self.num_samples).repeat(B, num_rays, 1)
        
        # Add uniform noise for continuous training (stratified sampling)
        if self.training:
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], -1)
            lower = torch.cat([t_vals[..., :1], mids], -1)
            t_rand = torch.rand(t_vals.shape, device=ro.device)
            t_vals = lower + (upper - lower) * t_rand

        # [B, H*W, num_samples, 3]
        pts = ro.unsqueeze(2) + rd.unsqueeze(2) * t_vals.unsqueeze(3)
        return pts, t_vals

    def query_volume(self, pts, volume):
        """
        Samples the 3D volume at specific continuous coordinates using grid_sample.
        Args:
            pts: [B, N_rays, N_samples, 3]
            volume: [B, C, Z, Y, X]
        """
        B, N_rays, N_samples, _ = pts.shape
        _, C, Z, Y, X = volume.shape
        
        # Normalize pts to [-1, 1] relative to volume bounds
        # Note grid_sample expects (X, Y, Z) ordering for 3D grids
        x_norm = (pts[..., 0] - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0]) * 2 - 1
        y_norm = (pts[..., 1] - self.y_bounds[0]) / (self.y_bounds[1] - self.y_bounds[0]) * 2 - 1
        z_norm = (pts[..., 2] - self.z_bounds[0]) / (self.z_bounds[1] - self.z_bounds[0]) * 2 - 1
        
        pts_norm = torch.stack([x_norm, y_norm, z_norm], dim=-1) # [B, N_rays, N_samples, 3]
        
        # Because grid_sample expects 5D input [B, N, D, H, W, 3] for 5D volume
        # We dummy reshape our rays into DxH
        pts_norm = pts_norm.unsqueeze(1) # [B, 1, N_rays, N_samples, 3]
        
        # sampled: [B, C, 1, N_rays, N_samples]
        sampled = F.grid_sample(volume, pts_norm, align_corners=True, padding_mode='zeros')
        
        # -> [B, N_rays, N_samples, C]
        sampled = sampled.squeeze(2).permute(0, 2, 3, 1)
        return sampled

    def render(self, H, W, K, T_v_c, semantic_volume, feature_volume):
        """
        Main entry point for Neural Volume Rendering.
        Converts the 3D volumes into 2D semantic/feature maps for the target camera.
        """
        B = K.shape[0]
        device = K.device
        
        # 1. Generate Rays
        ro, rd = self.get_rays(H, W, K, T_v_c)
        
        # 2. Sample Points along Rays (stratified)
        pts, t_vals = self.sample_along_rays(ro, rd)
        
        # 3. Query 3D Volumes at points
        # For simplicity in this dummy formulation, we treat the background class (0) 
        # or the logits magnitude as "density". 
        # Real formulations predict a dedicated 'density' (sigma) channel.
        # Let's mock a density by taking the max activation across semantic classes.
        
        sem_samples = self.query_volume(pts, semantic_volume) # [B, N_rays, N_samples, num_classes]
        feat_samples = self.query_volume(pts, feature_volume) # [B, N_rays, N_samples, feat_dim]
        
        # --- NeRF Alpha Compositing Equations ---
        # Get density (sigma)
        sigma = F.softplus(sem_samples.max(dim=-1)[0]) # [B, N_rays, N_samples]
        
        # Calculate distances between samples
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        # Append a large distance for the final sample
        dists = torch.cat([dists, torch.tensor([1e10], device=device).expand_as(dists[..., :1])], -1)
        
        # Alpha (opacity) -> 1 - exp(-sigma * delta)
        alpha = 1.0 - torch.exp(-sigma * dists)
        
        # Transmittance (cumulative product of transparencies)
        # Shift alphas to the right and pad with 1.0
        ones = torch.ones_like(alpha[..., :1])
        transmittance = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-10], -1), -1)[..., :-1]
        
        # Weights for each sample point along the ray
        weights = alpha * transmittance # [B, N_rays, N_samples]
        
        # Render (squash) by sum(weights * attributes)
        # 2D Semantic Feature Map
        weights_expanded = weights.unsqueeze(-1)
        rendered_sem = torch.sum(weights_expanded * sem_samples, dim=-2) # [B, N_rays, num_classes]
        rendered_sem = rendered_sem.transpose(1, 2).view(B, -1, H, W)    # [B, num_classes, H, W]
        
        # 2D DINOv2 Feature Map
        rendered_feat = torch.sum(weights_expanded * feat_samples, dim=-2) # [B, N_rays, feat_dim]
        rendered_feat = rendered_feat.transpose(1, 2).view(B, -1, H, W)    # [B, feat_dim, H, W]
        
        # Optional: Rendered Depth
        rendered_depth = torch.sum(weights * t_vals, dim=-1) # [B, N_rays]
        rendered_depth = rendered_depth.view(B, 1, H, W)     # [B, 1, H, W]
        
        return rendered_sem, rendered_feat, rendered_depth


if __name__ == "__main__":
    # Sanity Check
    B = 2
    H, W = 64, 128
    num_classes = 150
    feat_dim = 384
    
    # Mock Intrinsics & Extrinsics
    K = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    K[:, 0, 0] = 400.0
    K[:, 1, 1] = 400.0
    K[:, 0, 2] = W / 2.0
    K[:, 1, 2] = H / 2.0
    
    T_v_c = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    # Move camera up and back slightly
    T_v_c[:, 2, 3] = 1.5
    T_v_c[:, 0, 3] = -2.0
    
    # Mock Volumes from OffRoadOccNet output
    # (100x100x8 volume at 1m resolution -> bounds defined in init)
    semantic_volume = torch.randn(B, num_classes, 8, 100, 100)
    feature_volume = torch.randn(B, feat_dim, 8, 100, 100)
    
    renderer = VolumeRenderer(num_samples=32)
    
    print("Testing Volume Renderer forward pass...")
    sem_2d, feat_2d, depth_2d = renderer.render(H, W, K, T_v_c, semantic_volume, feature_volume)
    
    print(f"Rendered Semantics: {sem_2d.shape}")
    print(f"Rendered Features: {feat_2d.shape}")
    print(f"Rendered Depth: {depth_2d.shape}")
    print("Test passed successfully!")
