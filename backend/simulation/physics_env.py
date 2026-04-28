"""
Physics Environment using PyBullet.
Continuous 3D space with gravity, collisions, objects, and an embodied agent with a camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data


@dataclass
class EnvConfig:
    render_width: int = 128
    render_height: int = 128
    gravity: float = -9.81
    time_step: float = 1.0 / 240.0
    action_repeat: int = 4
    max_steps: int = 2000
    arena_size: float = 10.0
    num_cubes: int = 5
    num_targets: int = 2
    fov: float = 60.0
    near_plane: float = 0.1
    far_plane: float = 20.0
    agent_speed: float = 5.0
    agent_turn_speed: float = 3.0
    agent_height: float = 1.0
    agent_radius: float = 0.3


@dataclass
class AgentState:
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5]))
    orientation: float = 0.0  # yaw in radians
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_id: int = -1


@dataclass
class StepResult:
    observation: np.ndarray  # RGB frame (H, W, 3)
    reward: float
    done: bool
    info: dict[str, Any]


class PhysicsEnvironment:
    """Continuous 3D environment with PyBullet physics."""

    def __init__(self, config: EnvConfig | None = None, gui: bool = False):
        self.config = config or EnvConfig()
        self.gui = gui
        self.physics_client: int = -1
        self.agent = AgentState()
        self.objects: list[int] = []
        self.targets: list[int] = []
        self.target_positions: list[np.ndarray] = []
        self.step_count = 0
        self.current_target_idx = 0
        self.total_reward = 0.0
        self._injected_entities: list[dict[str, Any]] = []

    def reset(self) -> np.ndarray:
        if self.physics_client >= 0:
            p.disconnect(self.physics_client)

        mode = p.GUI if self.gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(0, 0, self.config.gravity, physicsClientId=self.physics_client)
        p.setTimeStep(self.config.time_step, physicsClientId=self.physics_client)

        self._create_ground()
        self._create_walls()
        self._create_agent()
        self._create_objects()
        self._create_targets()

        self.step_count = 0
        self.current_target_idx = 0
        self.total_reward = 0.0
        self._injected_entities.clear()

        for _ in range(10):
            p.stepSimulation(physicsClientId=self.physics_client)

        return self._get_camera_frame()

    def step(self, action: np.ndarray) -> StepResult:
        """
        Execute action in environment.
        action: [forward/backward, left/right, rotation] — continuous [-1, 1]
        """
        action = np.clip(action, -1.0, 1.0)
        forward = float(action[0]) * self.config.agent_speed
        strafe = float(action[1]) * self.config.agent_speed
        turn = float(action[2]) * self.config.agent_turn_speed

        self.agent.orientation += turn * self.config.time_step * self.config.action_repeat

        dx = forward * math.cos(self.agent.orientation) - strafe * math.sin(self.agent.orientation)
        dy = forward * math.sin(self.agent.orientation) + strafe * math.cos(self.agent.orientation)

        pos, orn = p.getBasePositionAndOrientation(
            self.agent.body_id, physicsClientId=self.physics_client
        )
        new_pos = [
            pos[0] + dx * self.config.time_step * self.config.action_repeat,
            pos[1] + dy * self.config.time_step * self.config.action_repeat,
            pos[2],
        ]

        quat = p.getQuaternionFromEuler([0, 0, self.agent.orientation])
        p.resetBasePositionAndOrientation(
            self.agent.body_id, new_pos, quat, physicsClientId=self.physics_client
        )

        for _ in range(self.config.action_repeat):
            p.stepSimulation(physicsClientId=self.physics_client)

        pos_after, _ = p.getBasePositionAndOrientation(
            self.agent.body_id, physicsClientId=self.physics_client
        )
        self.agent.position = np.array(pos_after)

        frame = self._get_camera_frame()
        reward = self._compute_reward()
        self.step_count += 1
        done = self.step_count >= self.config.max_steps
        self.total_reward += reward

        info = {
            "position": self.agent.position.tolist(),
            "orientation": self.agent.orientation,
            "step": self.step_count,
            "total_reward": self.total_reward,
            "target_idx": self.current_target_idx,
            "injected_entities": len(self._injected_entities),
        }

        return StepResult(observation=frame, reward=reward, done=done, info=info)

    def inject_entity(self, entity_type: str, position: np.ndarray | None = None,
                      metadata: dict[str, Any] | None = None) -> int:
        """Inject a new entity into the world (e.g., query_orb for questions)."""
        if position is None:
            agent_pos = self.agent.position.copy()
            forward_dir = np.array([
                math.cos(self.agent.orientation),
                math.sin(self.agent.orientation),
                0,
            ])
            position = agent_pos + forward_dir * 2.0
            position[2] = self.config.agent_height

        if entity_type == "query_orb":
            visual = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.3,
                rgbaColor=[0.2, 0.6, 1.0, 0.8],
                physicsClientId=self.physics_client,
            )
            collision = p.createCollisionShape(
                p.GEOM_SPHERE, radius=0.3,
                physicsClientId=self.physics_client,
            )
        else:
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2],
                rgbaColor=[1.0, 1.0, 0.0, 0.8],
                physicsClientId=self.physics_client,
            )
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2],
                physicsClientId=self.physics_client,
            )

        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position.tolist(),
            physicsClientId=self.physics_client,
        )

        entity_info = {
            "body_id": body_id,
            "type": entity_type,
            "position": position.tolist(),
            "metadata": metadata or {},
        }
        self._injected_entities.append(entity_info)
        return body_id

    def remove_entity(self, body_id: int) -> None:
        p.removeBody(body_id, physicsClientId=self.physics_client)
        self._injected_entities = [
            e for e in self._injected_entities if e["body_id"] != body_id
        ]

    def get_state(self) -> dict[str, Any]:
        pos, orn = p.getBasePositionAndOrientation(
            self.agent.body_id, physicsClientId=self.physics_client
        )
        vel, ang_vel = p.getBaseVelocity(
            self.agent.body_id, physicsClientId=self.physics_client
        )
        object_states = []
        for obj_id in self.objects + self.targets:
            obj_pos, obj_orn = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=self.physics_client
            )
            object_states.append({
                "id": obj_id,
                "position": list(obj_pos),
                "orientation": list(obj_orn),
            })

        return {
            "agent": {
                "position": list(pos),
                "orientation": self.agent.orientation,
                "velocity": list(vel),
                "angular_velocity": list(ang_vel),
            },
            "objects": object_states,
            "step": self.step_count,
            "total_reward": self.total_reward,
            "injected_entities": [
                {"type": e["type"], "position": e["position"]}
                for e in self._injected_entities
            ],
        }

    def close(self) -> None:
        if self.physics_client >= 0:
            p.disconnect(self.physics_client)
            self.physics_client = -1

    # --- Private methods ---

    def _create_ground(self) -> None:
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

    def _create_walls(self) -> None:
        half = self.config.arena_size / 2.0
        wall_height = 2.0
        wall_thickness = 0.2
        wall_color = [0.5, 0.5, 0.5, 1.0]

        walls = [
            ([0, half, wall_height / 2], [half, wall_thickness / 2, wall_height / 2]),
            ([0, -half, wall_height / 2], [half, wall_thickness / 2, wall_height / 2]),
            ([half, 0, wall_height / 2], [wall_thickness / 2, half, wall_height / 2]),
            ([-half, 0, wall_height / 2], [wall_thickness / 2, half, wall_height / 2]),
        ]

        for pos, extents in walls:
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=extents, rgbaColor=wall_color,
                physicsClientId=self.physics_client,
            )
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=extents,
                physicsClientId=self.physics_client,
            )
            p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual, basePosition=pos,
                physicsClientId=self.physics_client,
            )

    def _create_agent(self) -> None:
        visual = p.createVisualShape(
            p.GEOM_CAPSULE,
            radius=self.config.agent_radius,
            length=self.config.agent_height,
            rgbaColor=[0.1, 0.8, 0.2, 1.0],
            physicsClientId=self.physics_client,
        )
        collision = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=self.config.agent_radius,
            height=self.config.agent_height,
            physicsClientId=self.physics_client,
        )
        start_pos = [0, 0, self.config.agent_height / 2 + self.config.agent_radius]
        self.agent.body_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=start_pos,
            physicsClientId=self.physics_client,
        )
        self.agent.position = np.array(start_pos)
        self.agent.orientation = 0.0

    def _create_objects(self) -> None:
        rng = np.random.default_rng()
        colors = [
            [1, 0, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1],
            [1, 0, 1, 1], [0, 1, 1, 1],
        ]
        self.objects = []
        half = self.config.arena_size / 2.0 - 1.0
        for i in range(self.config.num_cubes):
            size = rng.uniform(0.2, 0.5)
            pos = [
                rng.uniform(-half, half),
                rng.uniform(-half, half),
                size,
            ]
            color = colors[i % len(colors)]
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[size, size, size],
                rgbaColor=color, physicsClientId=self.physics_client,
            )
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[size, size, size],
                physicsClientId=self.physics_client,
            )
            body_id = p.createMultiBody(
                baseMass=1.0, baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual, basePosition=pos,
                physicsClientId=self.physics_client,
            )
            self.objects.append(body_id)

    def _create_targets(self) -> None:
        rng = np.random.default_rng(42)
        self.targets = []
        self.target_positions = []
        half = self.config.arena_size / 2.0 - 1.0
        for _ in range(self.config.num_targets):
            pos = np.array([
                rng.uniform(-half, half),
                rng.uniform(-half, half),
                0.5,
            ])
            visual = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.4,
                rgbaColor=[0, 1, 0, 0.7],
                physicsClientId=self.physics_client,
            )
            collision = p.createCollisionShape(
                p.GEOM_SPHERE, radius=0.4,
                physicsClientId=self.physics_client,
            )
            body_id = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual, basePosition=pos.tolist(),
                physicsClientId=self.physics_client,
            )
            self.targets.append(body_id)
            self.target_positions.append(pos)

    def _get_camera_frame(self) -> np.ndarray:
        pos = self.agent.position
        yaw = self.agent.orientation
        eye_height = self.config.agent_height
        eye_pos = [pos[0], pos[1], eye_height]
        target_pos = [
            pos[0] + math.cos(yaw) * 2.0,
            pos[1] + math.sin(yaw) * 2.0,
            eye_height,
        ]

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.physics_client,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.config.fov,
            aspect=self.config.render_width / self.config.render_height,
            nearVal=self.config.near_plane,
            farVal=self.config.far_plane,
            physicsClientId=self.physics_client,
        )

        _, _, rgba, _, _ = p.getCameraImage(
            width=self.config.render_width,
            height=self.config.render_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.physics_client,
        )

        rgb = np.array(rgba, dtype=np.uint8).reshape(
            self.config.render_height, self.config.render_width, 4
        )[:, :, :3]
        return rgb

    def _compute_reward(self) -> float:
        reward = -0.01  # time penalty

        if self.current_target_idx < len(self.target_positions):
            target_pos = self.target_positions[self.current_target_idx]
            dist = np.linalg.norm(self.agent.position[:2] - target_pos[:2])
            if dist < 0.8:
                reward += 1.0
                self.current_target_idx += 1

        for entity in self._injected_entities:
            entity_pos = np.array(entity["position"])
            dist = np.linalg.norm(self.agent.position[:2] - entity_pos[:2])
            if dist < 1.0:
                reward += 0.5

        return reward
