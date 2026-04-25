"""
Procedural generation for per-user unique environments.
Uses Perlin noise for terrain, Poisson disk sampling for object placement,
and domain randomization for visual diversity.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class WorldSeed:
    """Deterministic seed derived from user + session IDs."""
    user_id: str
    session_id: str
    seed: int = 0

    def __post_init__(self):
        raw = f"{self.user_id}:{self.session_id}"
        self.seed = int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


@dataclass
class TerrainConfig:
    """Heightmap generation parameters."""
    scale: float = 4.0
    octaves: int = 4
    persistence: float = 0.5
    lacunarity: float = 2.0
    amplitude: float = 1.5
    resolution: int = 64


@dataclass
class ProceduralConfig:
    """Full procedural generation configuration."""
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    min_objects: int = 3
    max_objects: int = 12
    min_targets: int = 1
    max_targets: int = 4
    min_resources: int = 2
    max_resources: int = 8
    poisson_radius: float = 1.5
    domain_randomize: bool = True


def _perlin_gradient(seed: int, size: int) -> np.ndarray:
    """Generate a 2D gradient field for Perlin noise."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * math.pi, (size + 1, size + 1))
    gradients = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    return gradients


def _fade(t: np.ndarray) -> np.ndarray:
    """Smoothstep fade curve 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def generate_heightmap(
    seed: int,
    config: TerrainConfig,
    arena_size: float = 10.0,
) -> np.ndarray:
    """
    Generate a 2D heightmap using layered Perlin-like noise.
    Returns array of shape (resolution, resolution) with heights.
    """
    res = config.resolution
    heightmap = np.zeros((res, res), dtype=np.float32)

    for octave in range(config.octaves):
        freq = config.lacunarity ** octave
        amp = config.persistence ** octave * config.amplitude
        octave_seed = seed + octave * 7919

        grid_size = max(2, int(config.scale * freq))
        gradients = _perlin_gradient(octave_seed, grid_size)

        xs = np.linspace(0, grid_size - 1e-6, res)
        ys = np.linspace(0, grid_size - 1e-6, res)
        xg, yg = np.meshgrid(xs, ys)

        x0 = xg.astype(int)
        y0 = yg.astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        sx = _fade(xg - x0)
        sy = _fade(yg - y0)

        def dot_grid(gx: np.ndarray, gy: np.ndarray,
                     dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
            g = gradients[gy, gx]
            return g[..., 0] * dx + g[..., 1] * dy

        n00 = dot_grid(x0, y0, xg - x0, yg - y0)
        n10 = dot_grid(x1, y0, xg - x1, yg - y0)
        n01 = dot_grid(x0, y1, xg - x0, yg - y1)
        n11 = dot_grid(x1, y1, xg - x1, yg - y1)

        nx0 = _lerp(n00, n10, sx)
        nx1 = _lerp(n01, n11, sx)
        val = _lerp(nx0, nx1, sy)

        heightmap += val * amp

    hmin, hmax = heightmap.min(), heightmap.max()
    if hmax - hmin > 1e-6:
        heightmap = (heightmap - hmin) / (hmax - hmin)

    return heightmap


def poisson_disk_sampling(
    seed: int,
    arena_size: float,
    radius: float,
    max_points: int,
    margin: float = 1.0,
) -> list[tuple[float, float]]:
    """
    Generate well-distributed 2D points using Poisson disk sampling.
    Returns list of (x, y) positions within the arena.
    """
    rng = np.random.default_rng(seed)
    half = arena_size / 2.0 - margin
    cell_size = radius / math.sqrt(2)
    grid: dict[tuple[int, int], int] = {}

    points: list[tuple[float, float]] = []
    active: list[int] = []

    first = (rng.uniform(-half, half), rng.uniform(-half, half))
    points.append(first)
    active.append(0)
    gx, gy = int((first[0] + half) / cell_size), int((first[1] + half) / cell_size)
    grid[(gx, gy)] = 0

    k = 30

    while active and len(points) < max_points:
        idx = rng.integers(0, len(active))
        center = points[active[idx]]
        found = False

        for _ in range(k):
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(radius, 2 * radius)
            nx = center[0] + dist * math.cos(angle)
            ny = center[1] + dist * math.sin(angle)

            if nx < -half or nx > half or ny < -half or ny > half:
                continue

            gx = int((nx + half) / cell_size)
            gy = int((ny + half) / cell_size)

            too_close = False
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    key = (gx + dx, gy + dy)
                    if key in grid:
                        other = points[grid[key]]
                        d = math.sqrt((nx - other[0])**2 + (ny - other[1])**2)
                        if d < radius:
                            too_close = True
                            break
                if too_close:
                    break

            if not too_close:
                pidx = len(points)
                points.append((nx, ny))
                active.append(pidx)
                grid[(gx, gy)] = pidx
                found = True
                break

        if not found:
            active.pop(idx)

    return points


@dataclass
class DomainRandomization:
    """Randomized visual parameters for sim-to-real transfer."""
    object_colors: list[list[float]]
    wall_color: list[float]
    ground_color: list[float]
    sun_angle: float
    sun_intensity: float
    ambient_intensity: float
    fog_density: float


def generate_domain_randomization(seed: int) -> DomainRandomization:
    """Generate randomized visual parameters from a seed."""
    rng = np.random.default_rng(seed)

    num_colors = rng.integers(5, 12)
    colors = []
    for _ in range(num_colors):
        h = rng.uniform(0, 1)
        s = rng.uniform(0.5, 1.0)
        v = rng.uniform(0.4, 1.0)
        r, g, b = _hsv_to_rgb(h, s, v)
        colors.append([r, g, b, 1.0])

    wall_h = rng.uniform(0, 1)
    wall_color = list(_hsv_to_rgb(wall_h, 0.2, rng.uniform(0.3, 0.7))) + [1.0]

    ground_color = [
        rng.uniform(0.2, 0.5),
        rng.uniform(0.3, 0.6),
        rng.uniform(0.1, 0.4),
        1.0,
    ]

    return DomainRandomization(
        object_colors=colors,
        wall_color=wall_color,
        ground_color=ground_color,
        sun_angle=rng.uniform(15, 75),
        sun_intensity=rng.uniform(0.6, 1.2),
        ambient_intensity=rng.uniform(0.2, 0.5),
        fog_density=rng.uniform(0.0, 0.02),
    )


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV to RGB, all values in [0, 1]."""
    if s == 0:
        return (v, v, v)
    i = int(h * 6.0)
    f = h * 6.0 - i
    pp = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    if i == 0:
        return (v, t, pp)
    if i == 1:
        return (q, v, pp)
    if i == 2:
        return (pp, v, t)
    if i == 3:
        return (pp, q, v)
    if i == 4:
        return (t, pp, v)
    return (v, pp, q)


@dataclass
class WorldLayout:
    """Complete procedurally generated world layout."""
    seed: int
    heightmap: np.ndarray
    object_positions: list[tuple[float, float]]
    target_positions: list[tuple[float, float]]
    resource_positions: list[tuple[float, float]]
    domain_rand: DomainRandomization
    difficulty: float
    object_types: list[str]
    object_sizes: list[float]
    agent_spawn: tuple[float, float]


def generate_world(
    world_seed: WorldSeed,
    config: ProceduralConfig | None = None,
    arena_size: float = 10.0,
    difficulty: float = 0.5,
) -> WorldLayout:
    """
    Generate a complete world layout from a user-specific seed.
    Same seed always produces the same world.
    """
    cfg = config or ProceduralConfig()
    seed = world_seed.seed
    rng = np.random.default_rng(seed)

    heightmap = generate_heightmap(seed, cfg.terrain, arena_size)

    num_objects = rng.integers(cfg.min_objects, cfg.max_objects + 1)
    num_targets = rng.integers(cfg.min_targets, cfg.max_targets + 1)
    num_resources = rng.integers(cfg.min_resources, cfg.max_resources + 1)
    total_points = num_objects + num_targets + num_resources + 1

    all_positions = poisson_disk_sampling(
        seed=seed,
        arena_size=arena_size,
        radius=cfg.poisson_radius,
        max_points=total_points,
    )

    if not all_positions:
        all_positions = [(0.0, 0.0)]

    agent_spawn = all_positions[0]
    remaining = all_positions[1:]

    obj_count = min(num_objects, len(remaining))
    tgt_count = min(num_targets, len(remaining) - obj_count)
    res_count = min(num_resources, len(remaining) - obj_count - tgt_count)

    object_positions = remaining[:obj_count]
    target_positions = remaining[obj_count:obj_count + tgt_count]
    resource_positions = remaining[obj_count + tgt_count:obj_count + tgt_count + res_count]

    object_type_choices = ["cube", "cylinder", "sphere"]
    object_types = [object_type_choices[rng.integers(0, 3)] for _ in range(obj_count)]
    object_sizes = [float(rng.uniform(0.15, 0.5)) for _ in range(obj_count)]

    domain_rand = generate_domain_randomization(seed + 1000)

    return WorldLayout(
        seed=seed,
        heightmap=heightmap,
        object_positions=object_positions,
        target_positions=target_positions,
        resource_positions=resource_positions,
        domain_rand=domain_rand,
        difficulty=difficulty,
        object_types=object_types,
        object_sizes=object_sizes,
        agent_spawn=agent_spawn,
    )
