"""Tests for procedural generation."""

import numpy as np

from backend.simulation.procedural import (
    TerrainConfig,
    WorldSeed,
    generate_domain_randomization,
    generate_heightmap,
    generate_world,
    poisson_disk_sampling,
)


def test_world_seed_deterministic():
    """Same user+session always produces the same seed."""
    s1 = WorldSeed(user_id="user_1", session_id="session_a")
    s2 = WorldSeed(user_id="user_1", session_id="session_a")
    assert s1.seed == s2.seed


def test_world_seed_different_users():
    """Different users get different seeds."""
    s1 = WorldSeed(user_id="user_1", session_id="session_a")
    s2 = WorldSeed(user_id="user_2", session_id="session_a")
    assert s1.seed != s2.seed


def test_heightmap_generation():
    """Heightmap has correct shape and is normalized to [0, 1]."""
    config = TerrainConfig(resolution=32, octaves=2)
    hmap = generate_heightmap(seed=42, config=config)
    assert hmap.shape == (32, 32)
    assert hmap.min() >= 0.0
    assert hmap.max() <= 1.0


def test_heightmap_reproducible():
    """Same seed produces identical heightmaps."""
    config = TerrainConfig(resolution=16, octaves=2)
    h1 = generate_heightmap(seed=123, config=config)
    h2 = generate_heightmap(seed=123, config=config)
    np.testing.assert_array_equal(h1, h2)


def test_poisson_disk_sampling():
    """Poisson disk sampling produces well-spaced points."""
    points = poisson_disk_sampling(seed=42, arena_size=10.0, radius=1.5, max_points=20)
    assert len(points) > 0
    assert len(points) <= 20
    # Check minimum spacing
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i != j:
                dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
                assert dist >= 1.5 - 0.01  # small tolerance


def test_domain_randomization():
    """Domain randomization produces valid colors and parameters."""
    dr = generate_domain_randomization(seed=42)
    assert len(dr.object_colors) >= 5
    for color in dr.object_colors:
        assert len(color) == 4
        assert all(0 <= c <= 1 for c in color)
    assert 15 <= dr.sun_angle <= 75
    assert 0.6 <= dr.sun_intensity <= 1.2


def test_generate_world():
    """Full world generation produces complete layout."""
    ws = WorldSeed(user_id="test_user", session_id="test_session")
    layout = generate_world(ws, arena_size=10.0)
    assert layout.seed == ws.seed
    assert layout.heightmap.shape[0] > 0
    assert len(layout.object_positions) >= 0
    assert len(layout.target_positions) >= 0
    assert len(layout.resource_positions) >= 0
    assert layout.domain_rand is not None


def test_generate_world_reproducible():
    """Same seed produces same world layout."""
    ws = WorldSeed(user_id="u1", session_id="s1")
    w1 = generate_world(ws, arena_size=10.0)
    w2 = generate_world(ws, arena_size=10.0)
    assert w1.seed == w2.seed
    np.testing.assert_array_equal(w1.heightmap, w2.heightmap)
    assert w1.object_positions == w2.object_positions
    assert w1.target_positions == w2.target_positions
