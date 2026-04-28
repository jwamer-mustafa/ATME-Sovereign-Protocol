"""
Ecology Layer: Dynamic environment with resources, NPCs, weather events,
day/night cycles, and adaptive difficulty.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class WeatherType(str, Enum):
    CLEAR = "clear"
    RAIN = "rain"
    STORM = "storm"
    FOG = "fog"


class TimeOfDay(str, Enum):
    DAWN = "dawn"
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"


@dataclass
class Resource:
    """Collectible resource in the environment."""
    resource_id: str
    resource_type: str
    position: list[float]
    body_id: int = -1
    value: float = 1.0
    respawn_time: float = 30.0
    collected: bool = False
    collected_at: float = 0.0


@dataclass
class NPC:
    """Non-player character with simple behavior."""
    npc_id: str
    behavior: str  # "patrol", "wander", "follow", "flee"
    position: list[float]
    body_id: int = -1
    speed: float = 1.0
    patrol_points: list[list[float]] = field(default_factory=list)
    current_patrol_idx: int = 0
    orientation: float = 0.0


@dataclass
class WeatherEvent:
    """Active weather event."""
    weather_type: WeatherType
    intensity: float
    start_time: float
    duration: float
    wind_direction: list[float] = field(default_factory=lambda: [1.0, 0.0])
    wind_speed: float = 0.0


@dataclass
class DayNightCycle:
    """Day/night cycle state."""
    time_scale: float = 60.0  # sim seconds per day cycle
    current_time: float = 0.25  # 0-1, 0.25 = dawn
    sun_angle: float = 45.0
    sun_intensity: float = 1.0
    ambient_intensity: float = 0.3
    sky_color: list[float] = field(default_factory=lambda: [0.5, 0.7, 1.0])

    def update(self, dt: float) -> None:
        """Advance the day/night cycle."""
        self.current_time = (self.current_time + dt / self.time_scale) % 1.0

        self.sun_angle = self.current_time * 360.0 - 90.0
        sun_elevation = math.sin(self.current_time * math.pi)

        self.sun_intensity = max(0.05, sun_elevation)
        self.ambient_intensity = 0.1 + 0.3 * max(0, sun_elevation)

        if sun_elevation > 0.3:
            self.sky_color = [0.5, 0.7, 1.0]
        elif sun_elevation > 0:
            t = sun_elevation / 0.3
            self.sky_color = [
                0.8 - 0.3 * t,
                0.3 + 0.4 * t,
                0.3 + 0.7 * t,
            ]
        else:
            self.sky_color = [0.05, 0.05, 0.15]

    @property
    def time_of_day(self) -> TimeOfDay:
        if 0.2 <= self.current_time < 0.3:
            return TimeOfDay.DAWN
        elif 0.3 <= self.current_time < 0.7:
            return TimeOfDay.DAY
        elif 0.7 <= self.current_time < 0.8:
            return TimeOfDay.DUSK
        return TimeOfDay.NIGHT


@dataclass
class CurriculumState:
    """Adaptive difficulty state."""
    difficulty: float = 0.3
    min_difficulty: float = 0.1
    max_difficulty: float = 1.0
    success_window: list[bool] = field(default_factory=list)
    window_size: int = 20
    target_success_rate: float = 0.6
    adjustment_rate: float = 0.05

    def record_outcome(self, success: bool) -> None:
        """Record a task outcome and adjust difficulty."""
        self.success_window.append(success)
        if len(self.success_window) > self.window_size:
            self.success_window = self.success_window[-self.window_size:]

        if len(self.success_window) >= self.window_size // 2:
            rate = sum(self.success_window) / len(self.success_window)
            if rate > self.target_success_rate + 0.1:
                self.difficulty = min(
                    self.max_difficulty,
                    self.difficulty + self.adjustment_rate,
                )
            elif rate < self.target_success_rate - 0.1:
                self.difficulty = max(
                    self.min_difficulty,
                    self.difficulty - self.adjustment_rate,
                )


class EcologyManager:
    """Manages the living ecosystem: resources, NPCs, weather, day/night."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.resources: dict[str, Resource] = {}
        self.npcs: dict[str, NPC] = {}
        self.weather: WeatherEvent | None = None
        self.day_night = DayNightCycle()
        self.curriculum = CurriculumState()
        self._resource_counter = 0
        self._npc_counter = 0
        self._weather_timer = 0.0
        self._weather_interval = 120.0

    def spawn_resource(
        self,
        position: list[float],
        resource_type: str = "energy",
        value: float = 1.0,
    ) -> Resource:
        """Spawn a collectible resource."""
        self._resource_counter += 1
        rid = f"res_{self._resource_counter}"
        resource = Resource(
            resource_id=rid,
            resource_type=resource_type,
            position=position,
            value=value,
        )
        self.resources[rid] = resource
        return resource

    def spawn_npc(
        self,
        position: list[float],
        behavior: str = "wander",
        patrol_points: list[list[float]] | None = None,
    ) -> NPC:
        """Spawn an NPC with a specified behavior."""
        self._npc_counter += 1
        nid = f"npc_{self._npc_counter}"
        npc = NPC(
            npc_id=nid,
            behavior=behavior,
            position=position,
            patrol_points=patrol_points or [],
        )
        self.npcs[nid] = npc
        return npc

    def update(self, dt: float, agent_position: list[float],
               current_time: float) -> dict[str, Any]:
        """
        Update all ecology systems. Returns events that occurred.
        """
        events: list[dict[str, Any]] = []

        self.day_night.update(dt)

        self._update_npcs(dt, agent_position)

        collected = self._check_resource_collection(agent_position)
        for res in collected:
            events.append({
                "type": "resource_collected",
                "resource_id": res.resource_id,
                "resource_type": res.resource_type,
                "value": res.value,
            })

        self._respawn_resources(current_time)

        self._weather_timer += dt
        if self._weather_timer >= self._weather_interval:
            self._weather_timer = 0.0
            weather_event = self._maybe_change_weather(current_time)
            if weather_event:
                events.append({
                    "type": "weather_change",
                    "weather": weather_event.weather_type.value,
                    "intensity": weather_event.intensity,
                })

        return {
            "events": events,
            "day_night": {
                "time": self.day_night.current_time,
                "time_of_day": self.day_night.time_of_day.value,
                "sun_angle": self.day_night.sun_angle,
                "sun_intensity": self.day_night.sun_intensity,
                "ambient_intensity": self.day_night.ambient_intensity,
                "sky_color": self.day_night.sky_color,
            },
            "weather": {
                "type": self.weather.weather_type.value if self.weather else "clear",
                "intensity": self.weather.intensity if self.weather else 0.0,
                "wind_direction": (
                    self.weather.wind_direction if self.weather
                    else [0.0, 0.0]
                ),
                "wind_speed": self.weather.wind_speed if self.weather else 0.0,
            },
            "difficulty": self.curriculum.difficulty,
        }

    def get_state(self) -> dict[str, Any]:
        """Get current ecology state for streaming."""
        return {
            "resources": [
                {
                    "id": r.resource_id,
                    "type": r.resource_type,
                    "position": r.position,
                    "collected": r.collected,
                    "value": r.value,
                }
                for r in self.resources.values()
            ],
            "npcs": [
                {
                    "id": n.npc_id,
                    "behavior": n.behavior,
                    "position": n.position,
                    "orientation": n.orientation,
                }
                for n in self.npcs.values()
            ],
            "day_night": {
                "time": self.day_night.current_time,
                "time_of_day": self.day_night.time_of_day.value,
                "sun_intensity": self.day_night.sun_intensity,
                "sky_color": self.day_night.sky_color,
            },
            "weather": {
                "type": self.weather.weather_type.value if self.weather else "clear",
                "intensity": self.weather.intensity if self.weather else 0.0,
            },
            "difficulty": self.curriculum.difficulty,
        }

    def _update_npcs(self, dt: float, agent_position: list[float]) -> None:
        """Update NPC positions based on their behavior."""
        for npc in self.npcs.values():
            if npc.behavior == "patrol" and npc.patrol_points:
                target = npc.patrol_points[npc.current_patrol_idx]
                dx = target[0] - npc.position[0]
                dy = target[1] - npc.position[1]
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < 0.5:
                    npc.current_patrol_idx = (
                        (npc.current_patrol_idx + 1) % len(npc.patrol_points)
                    )
                elif dist > 0:
                    npc.position[0] += (dx / dist) * npc.speed * dt
                    npc.position[1] += (dy / dist) * npc.speed * dt
                    npc.orientation = math.atan2(dy, dx)

            elif npc.behavior == "wander":
                npc.orientation += self.rng.uniform(-0.5, 0.5) * dt
                npc.position[0] += math.cos(npc.orientation) * npc.speed * dt * 0.3
                npc.position[1] += math.sin(npc.orientation) * npc.speed * dt * 0.3

                half = 4.0
                npc.position[0] = max(-half, min(half, npc.position[0]))
                npc.position[1] = max(-half, min(half, npc.position[1]))

            elif npc.behavior == "follow":
                dx = agent_position[0] - npc.position[0]
                dy = agent_position[1] - npc.position[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 2.0:
                    npc.position[0] += (dx / dist) * npc.speed * dt
                    npc.position[1] += (dy / dist) * npc.speed * dt
                    npc.orientation = math.atan2(dy, dx)

            elif npc.behavior == "flee":
                dx = npc.position[0] - agent_position[0]
                dy = npc.position[1] - agent_position[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 5.0 and dist > 0:
                    npc.position[0] += (dx / dist) * npc.speed * dt * 1.5
                    npc.position[1] += (dy / dist) * npc.speed * dt * 1.5
                    npc.orientation = math.atan2(dy, dx)

    def _check_resource_collection(
        self, agent_position: list[float],
    ) -> list[Resource]:
        """Check if agent is close enough to collect resources."""
        collected = []
        for resource in self.resources.values():
            if resource.collected:
                continue
            dx = agent_position[0] - resource.position[0]
            dy = agent_position[1] - resource.position[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.8:
                resource.collected = True
                resource.collected_at = time.time()
                collected.append(resource)
        return collected

    def _respawn_resources(self, current_time: float) -> None:
        """Respawn collected resources after their respawn timer."""
        for resource in self.resources.values():
            if resource.collected:
                if current_time - resource.collected_at >= resource.respawn_time:
                    resource.collected = False

    def _maybe_change_weather(self, current_time: float) -> WeatherEvent | None:
        """Stochastically change weather (Poisson process)."""
        if self.rng.random() < 0.3:
            weather_type = self.rng.choice([
                WeatherType.CLEAR,
                WeatherType.RAIN,
                WeatherType.STORM,
                WeatherType.FOG,
            ])
            intensity = float(self.rng.uniform(0.3, 1.0))
            wind_angle = float(self.rng.uniform(0, 2 * math.pi))

            self.weather = WeatherEvent(
                weather_type=weather_type,
                intensity=intensity,
                start_time=current_time,
                duration=float(self.rng.uniform(30, 120)),
                wind_direction=[math.cos(wind_angle), math.sin(wind_angle)],
                wind_speed=float(self.rng.uniform(0, 5.0)) * intensity,
            )
            return self.weather
        return None

    def get_reward_modifier(self) -> float:
        """Get reward modifier based on ecology conditions."""
        modifier = 1.0

        if self.day_night.time_of_day == TimeOfDay.NIGHT:
            modifier *= 1.5

        if self.weather and self.weather.weather_type == WeatherType.STORM:
            modifier *= 1.3

        modifier *= 1.0 + self.curriculum.difficulty * 0.5

        return modifier
