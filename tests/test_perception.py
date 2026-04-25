"""Tests for the perception pipeline."""

import torch

from backend.perception.encoder import PerceptionPipeline, SaliencyExtractor, VisionEncoder


def test_vision_encoder():
    encoder = VisionEncoder(input_channels=3, latent_dim=256, input_size=128)
    x = torch.randn(2, 3, 128, 128)
    z, feature_maps = encoder(x)
    assert z.shape == (2, 256)
    assert len(feature_maps.shape) == 4


def test_saliency_simple():
    feature_maps = torch.randn(2, 128, 8, 8)
    attention = SaliencyExtractor.simple_attention(feature_maps, target_size=(128, 128))
    assert attention.shape == (2, 1, 128, 128)
    assert attention.min() >= 0
    assert attention.max() <= 1.0 + 1e-6


def test_perception_pipeline():
    pipeline = PerceptionPipeline(latent_dim=256, input_size=128)
    frame = torch.randn(1, 3, 128, 128)
    result = pipeline.process_frame(frame)
    assert "latent" in result
    assert "attention_map" in result
    assert "features_downsampled" in result
    assert result["latent"].shape == (1, 256)
    assert result["attention_map"].shape == (1, 1, 128, 128)
