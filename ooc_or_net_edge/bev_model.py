"""
bev_model.py
BEV prediction model adapted for 100×100 FOV-limited grid.

Architecture:
  EfficientNet-B0 backbone → DepthHead → LSS View Transform → BEV Decoder
  Grid: [-12.5, 12.5] lateral × [0, 25] forward, 0.25m resolution → 100×100

Outputs:
  - semantic_logits: [B, 13, 100, 100]
  - height_pred:     [B, 1, 100, 100]
  - slope_pred:      [B, 1, 100, 100]
  - depths_2d:       [B, 1, fH, fW]  (auxiliary depth prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── Backbone ─────────────────────────────────────────────────────────

class EfficientNetBackbone(nn.Module):
    """EfficientNet-B0 feature extractor at 1/16 scale."""

    def __init__(self, out_channels=128):
        super().__init__()
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(effnet.features.children())[:6])
        self.conv_out = nn.Sequential(
            nn.Conv2d(112, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_out(self.backbone(x))  # [B, C, H/16, W/16]


# ── Auxiliary Depth Head ─────────────────────────────────────────────

class DepthHead(nn.Module):
    """Predicts per-pixel absolute depth from backbone features."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        return torch.relu(self.conv(x))  # [B, 1, fH, fW], positive depth


# ── LSS View Transformer ────────────────────────────────────────────

class LSSViewTransformer(nn.Module):
    """
    Deterministic Lift-Splat-Shoot.
    Lifts 2D features to 3D via predicted depth, then splatters into 2D BEV grid.

    Grid configuration for our 100×100 BEV:
      - X (lateral): [-12.5, 12.5] m  → 100 cells
      - Y (forward): [0, 25] m        → 100 cells
      - Z (height):  [-2.0, 3.0] m    → collapsed

    In the camera frame: X_cam = lateral, Y_cam = down, Z_cam = forward.
    We map: BEV_col ← X_cam, BEV_row ← Z_cam.
    """

    def __init__(
        self,
        x_bounds=(-12.5, 12.5),
        y_bounds=(0.0, 25.0),
        z_bounds=(-2.0, 3.0),
        resolution=0.25,
    ):
        super().__init__()
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.z_min, self.z_max = z_bounds
        self.res = resolution

        self.nx = int((self.x_max - self.x_min) / resolution)  # 100
        self.ny = int((self.y_max - self.y_min) / resolution)  # 100

    def forward(self, features, depths, intrinsics, extrinsics):
        """
        features:    [B, C, fH, fW]
        depths:      [B, 1, fH, fW]  absolute depth in metres
        intrinsics:  [B, 4, 4]       at feature-map scale
        extrinsics:  [B, 4, 4]       T_veh_cam (vehicle/lidar → camera)
        Returns:     [B, C, ny, nx]  (ny=100 forward rows, nx=100 lateral cols)
        """
        B, C, fH, fW = features.shape
        device = features.device

        # Pixel grid
        yy, xx = torch.meshgrid(
            torch.arange(fH, device=device, dtype=torch.float32),
            torch.arange(fW, device=device, dtype=torch.float32),
            indexing="ij",
        )
        pix = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.ones(fH * fW, device=device)], dim=0)  # [3, N]

        bev = torch.zeros((B, C, self.ny, self.nx), device=device)

        for b in range(B):
            K_inv = torch.inverse(intrinsics[b, :3, :3])
            T_vc = extrinsics[b]  # T_veh_cam: maps vehicle points → camera frame
            # We need camera → vehicle, so invert:
            T_cv = torch.inverse(T_vc)

            depth_flat = depths[b].view(1, -1)  # [1, N]
            valid = (depth_flat > 0.5) & (depth_flat < 50.0)
            valid = valid.squeeze(0)  # [N]

            rays = K_inv @ pix  # [3, N]
            pts_cam = rays[:, valid] * depth_flat[0, valid].unsqueeze(0)  # [3, Nv]

            if pts_cam.shape[1] == 0:
                continue

            # Camera → Vehicle frame
            pts_hom = torch.cat([pts_cam, torch.ones(1, pts_cam.shape[1], device=device)], dim=0)  # [4, Nv]
            pts_veh = (T_cv @ pts_hom)[:3]  # [3, Nv]

            # In camera frame: X_cam = lateral, Z_cam = forward, Y_cam = down
            # Our extrinsic maps vehicle→camera, so after inverting:
            # pts_veh is in the vehicle/lidar frame.
            # But our BEV grid is defined in camera frame coords:
            #   BEV lateral (col) = X_cam, BEV forward (row) = Z_cam
            # So we actually project in camera frame directly:
            x_cam = pts_cam[0]  # lateral
            z_cam = pts_cam[2]  # forward
            y_cam = pts_cam[1]  # height (down)

            # Height filter
            h_valid = (y_cam >= self.z_min) & (y_cam <= self.z_max)
            # BEV bounds
            bev_valid = (
                h_valid
                & (x_cam >= self.x_min) & (x_cam < self.x_max)
                & (z_cam >= self.y_min) & (z_cam < self.y_max)
            )

            x_f = x_cam[bev_valid]
            z_f = z_cam[bev_valid]

            if x_f.shape[0] == 0:
                continue

            # Grid indices
            col = torch.clamp(((x_f - self.x_min) / self.res).long(), 0, self.nx - 1)
            row = torch.clamp(((z_f - self.y_min) / self.res).long(), 0, self.ny - 1)
            # Flip row so row-0 = farthest (matching target convention)
            row = (self.ny - 1) - row

            # Gather features for valid points
            feat_flat = features[b].view(C, -1)  # [C, N_all]
            valid_indices = torch.where(valid)[0]
            valid_bev_indices = valid_indices[bev_valid]
            feat_sel = feat_flat[:, valid_bev_indices]  # [C, N_bev]

            # Scatter-add into grid
            flat_idx = (row * self.nx + col).unsqueeze(0).expand(C, -1)  # [C, N_bev]
            bev_flat = torch.zeros(C, self.ny * self.nx, device=device)
            bev_flat.scatter_add_(1, flat_idx, feat_sel)

            # Hit count for averaging
            hit_flat = torch.zeros(1, self.ny * self.nx, device=device)
            hit_flat.scatter_add_(1, flat_idx[0:1], torch.ones(1, feat_sel.shape[1], device=device))
            bev_flat = bev_flat / hit_flat.clamp(min=1.0)

            bev[b] = bev_flat.view(C, self.ny, self.nx)

        return bev


# ── BEV Decoder Head ─────────────────────────────────────────────────

class BEVDecoderHead(nn.Module):
    """
    Multi-task BEV decoder with three branches:
      1. Semantic segmentation (13 classes)
      2. Height regression
      3. Slope regression
    """

    def __init__(self, in_channels, num_classes=13):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.sem_head = nn.Conv2d(64, num_classes, 1)
        self.height_head = nn.Conv2d(64, 1, 1)
        self.slope_head = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True),  # slope ≥ 0
        )

    def forward(self, x):
        feat = self.shared(x)
        return self.sem_head(feat), self.height_head(feat), self.slope_head(feat)


# ── Full Model ───────────────────────────────────────────────────────

class BEVPredictionNet(nn.Module):
    """
    RGB → BEV prediction network.

    Pipeline: EfficientNet-B0 → DepthHead → LSS → BEV Decoder
    No temporal fusion (single-frame).
    """

    def __init__(self, num_classes=13, embed_dim=128):
        super().__init__()
        self.backbone = EfficientNetBackbone(out_channels=embed_dim)
        self.depth_head = DepthHead(in_channels=embed_dim)
        self.lss = LSSViewTransformer(
            x_bounds=(-12.5, 12.5),
            y_bounds=(0.0, 25.0),
            z_bounds=(-2.0, 3.0),
            resolution=0.25,
        )
        self.decoder = BEVDecoderHead(in_channels=embed_dim, num_classes=num_classes)

    def forward(self, images, intrinsics, extrinsics):
        """
        Args:
            images:      [B, 3, H, W]
            intrinsics:  [B, 4, 4]  (at original image scale)
            extrinsics:  [B, 4, 4]  T_veh_cam
        Returns:
            sem_logits:  [B, 13, 100, 100]
            height_pred: [B, 1, 100, 100]
            slope_pred:  [B, 1, 100, 100]
            depths_2d:   [B, 1, fH, fW]
        """
        # Backbone
        features = self.backbone(images)  # [B, C, H/16, W/16]

        # Auxiliary depth
        depths_2d = self.depth_head(features)  # [B, 1, fH, fW]

        # Scale intrinsics to feature map resolution (1/16)
        scale = 1.0 / 16.0
        K_scaled = intrinsics.clone()
        K_scaled[:, 0, :] *= scale  # fx, cx
        K_scaled[:, 1, :] *= scale  # fy, cy

        # LSS: 2D features → BEV grid
        bev_features = self.lss(features, depths_2d, K_scaled, extrinsics)  # [B, C, 100, 100]

        # Decode
        sem_logits, height_pred, slope_pred = self.decoder(bev_features)

        return sem_logits, height_pred, slope_pred, depths_2d


# ── Quick test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    model = BEVPredictionNet(num_classes=13, embed_dim=128)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.2f} M")

    B = 2
    images = torch.randn(B, 3, 360, 640)
    K = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    K[:, 0, 0] = 518.7  # fx scaled
    K[:, 1, 1] = 374.9  # fy scaled
    K[:, 0, 2] = 314.4  # cx scaled
    K[:, 1, 2] = 183.3  # cy scaled
    T = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)

    t0 = time.time()
    with torch.no_grad():
        sem, h, s, d = model(images, K, T)
    print(f"Forward: {time.time() - t0:.3f}s")
    print(f"Semantic: {sem.shape}")
    print(f"Height:   {h.shape}")
    print(f"Slope:    {s.shape}")
    print(f"Depth 2D: {d.shape}")
