"""
Vision Encoder: CNN-based feature extraction with attention/saliency maps.
Pipeline: RGB Frame → Encoder (CNN) → Latent (z) → Policy/Value + Attention Map
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VisionEncoder(nn.Module):
    """Lightweight CNN encoder: Conv→ReLU→Pool ×3 → latent z."""

    def __init__(self, input_channels: int = 3, latent_dim: int = 256,
                 input_size: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        test_input = torch.zeros(1, input_channels, input_size, input_size)
        with torch.no_grad():
            test_output = self._forward_conv(test_input)
        flat_size = test_output.shape[1] * test_output.shape[2] * test_output.shape[3]

        self.fc = nn.Linear(flat_size, latent_dim)
        self._feature_map_shape = test_output.shape[1:]

    def _forward_conv(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (latent_z, feature_maps) for downstream use."""
        feature_maps = self._forward_conv(x)
        flat = feature_maps.flatten(start_dim=1)
        z = self.fc(flat)
        return z, feature_maps

    def get_feature_map_shape(self) -> tuple[int, ...]:
        return tuple(self._feature_map_shape)


class SaliencyExtractor:
    """Grad-CAM style saliency extraction from feature maps."""

    @staticmethod
    def grad_cam(feature_maps: Tensor, gradients: Tensor,
                 target_size: tuple[int, int] = (128, 128)) -> Tensor:
        """
        Compute Grad-CAM saliency map.
        feature_maps: (B, C, H, W)
        gradients: (B, C, H, W) — gradients of target w.r.t. feature maps
        Returns: (B, 1, target_H, target_W) normalized heatmap
        """
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)

        # Normalize per sample
        b = cam.shape[0]
        cam_flat = cam.view(b, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1)
        cam_max = cam_flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    @staticmethod
    def simple_attention(feature_maps: Tensor,
                         target_size: tuple[int, int] = (128, 128)) -> Tensor:
        """Simple channel-mean attention map (no gradients needed)."""
        attention = feature_maps.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        attention = F.interpolate(
            attention, size=target_size, mode="bilinear", align_corners=False
        )
        b = attention.shape[0]
        att_flat = attention.view(b, -1)
        att_min = att_flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1)
        att_max = att_flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
        attention = (attention - att_min) / (att_max - att_min + 1e-8)
        return attention


class PerceptionPipeline:
    """Full perception pipeline: frame → encoder → latent + attention + metrics."""

    def __init__(self, latent_dim: int = 256, input_size: int = 128,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.encoder = VisionEncoder(
            latent_dim=latent_dim, input_size=input_size
        ).to(self.device)
        self.saliency = SaliencyExtractor()
        self.input_size = input_size

    def process_frame(self, frame_rgb: Tensor) -> dict[str, Tensor]:
        """
        Process a single frame through the perception pipeline.
        frame_rgb: (B, 3, H, W) float tensor [0, 1]
        Returns dict with latent, attention_map, features, etc.
        """
        self.encoder.eval()
        frame_rgb = frame_rgb.to(self.device)

        with torch.no_grad():
            z, feature_maps = self.encoder(frame_rgb)
            attention_map = self.saliency.simple_attention(
                feature_maps, target_size=(self.input_size, self.input_size)
            )

        features_downsampled = F.adaptive_avg_pool2d(feature_maps, (8, 8))

        return {
            "latent": z,
            "feature_maps": feature_maps,
            "attention_map": attention_map,
            "features_downsampled": features_downsampled,
        }

    def process_frame_with_gradients(self, frame_rgb: Tensor,
                                     target_fn: callable) -> dict[str, Tensor]:
        """
        Process with Grad-CAM saliency (requires a differentiable target).
        target_fn: callable that takes latent z and returns scalar target.
        """
        self.encoder.eval()
        frame_rgb = frame_rgb.to(self.device).requires_grad_(False)

        z, feature_maps = self.encoder(frame_rgb)
        feature_maps.retain_grad()
        target = target_fn(z)
        target.backward(retain_graph=True)

        grad_cam_map = self.saliency.grad_cam(
            feature_maps, feature_maps.grad,
            target_size=(self.input_size, self.input_size),
        )

        return {
            "latent": z.detach(),
            "feature_maps": feature_maps.detach(),
            "attention_map": grad_cam_map.detach(),
            "features_downsampled": F.adaptive_avg_pool2d(
                feature_maps.detach(), (8, 8)
            ),
        }
