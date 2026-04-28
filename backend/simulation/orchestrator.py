"""
Simulation Orchestrator: Ties together physics, perception, learning, memory,
event injection, procedural generation, ecology, multilingual, and retention
into a coherent simulation loop.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import torch

from backend.ethics.safeguards import format_ethical_response
from backend.learning.ppo import PPOConfig, PPOTrainer
from backend.memory.episodic import EpisodicMemory, ReplayBuffer
from backend.multilingual.translator import MultilingualPipeline
from backend.perception.encoder import PerceptionPipeline
from backend.retention.user_memory import RetentionManager
from backend.simulation.ecology import EcologyManager
from backend.simulation.event_injector import (
    EventInjector,
    ResponseGenerator,
    get_orb_animation_state,
)
from backend.simulation.physics_env import EnvConfig, PhysicsEnvironment, StepResult
from backend.simulation.procedural import (
    ProceduralConfig,
    WorldSeed,
    generate_world,
)
from backend.streaming.ws_manager import (
    ConnectionManager,
    prepare_perception_payload,
)


class SimulationOrchestrator:
    """Main orchestrator that runs the simulation loop."""

    def __init__(self, env_config: EnvConfig | None = None,
                 ppo_config: PPOConfig | None = None,
                 device: str = "cpu",
                 user_id: str = "anon",
                 session_id: str = "default"):
        self.env = PhysicsEnvironment(config=env_config)
        self.perception = PerceptionPipeline(device=device)
        self.trainer = PPOTrainer(config=ppo_config, device=device)
        self.memory = EpisodicMemory(capacity=10000)
        self.replay_buffer = ReplayBuffer()
        self.event_injector = EventInjector()
        self.response_generator = ResponseGenerator()
        self.ws_manager = ConnectionManager()

        # Phase B modules
        self.ecology = EcologyManager()
        self.multilingual = MultilingualPipeline()
        self.retention = RetentionManager()
        self.user_id = user_id
        self.session_id = session_id

        # Procedural generation
        self.world_seed = WorldSeed(user_id=user_id, session_id=session_id)
        self.procedural_config = ProceduralConfig()
        self.world_layout = None

        self.device = device
        self.running = False
        self.paused = False
        self.speed_multiplier = 1.0
        self.step_count = 0
        self.episode_count = 0
        self.current_latent: np.ndarray | None = None
        self.current_action_info: dict[str, Any] = {}
        self._training_enabled = True

    def reset(self) -> np.ndarray:
        """Reset the environment, apply procedural gen, and spawn ecology."""
        frame = self.env.reset()
        self.step_count = 0
        self.episode_count += 1

        # Generate procedural world layout
        self.world_layout = generate_world(
            self.world_seed,
            config=self.procedural_config,
            arena_size=self.env.config.arena_size,
            difficulty=self.ecology.curriculum.difficulty,
        )

        # Spawn ecology resources at procedurally-placed positions
        self.ecology = EcologyManager(seed=self.world_seed.seed)
        resource_types = ["energy", "food", "points"]
        for i, pos in enumerate(self.world_layout.resource_positions):
            rtype = resource_types[i % len(resource_types)]
            self.ecology.spawn_resource(
                position=[pos[0], pos[1], 0.3],
                resource_type=rtype,
                value=1.0 + self.ecology.curriculum.difficulty,
            )

        # Spawn NPCs
        behaviors = ["wander", "patrol", "follow", "flee"]
        npc_positions = self.world_layout.object_positions[:3]
        for i, pos in enumerate(npc_positions):
            self.ecology.spawn_npc(
                position=[pos[0], pos[1], 0.5],
                behavior=behaviors[i % len(behaviors)],
            )

        return frame

    def step(self, frame: np.ndarray) -> dict[str, Any]:
        """Run one simulation step: perceive → decide → act → learn."""
        frame_tensor = (
            torch.FloatTensor(frame).permute(2, 0, 1).unsqueeze(0) / 255.0
        )

        perception = self.perception.process_frame(frame_tensor)
        latent_z = perception["latent"].cpu().numpy().squeeze(0)
        self.current_latent = latent_z

        action_info = self.trainer.select_action(latent_z)
        self.current_action_info = action_info
        action = action_info["action"]

        result = self.env.step(action)

        if self._training_enabled:
            self.trainer.store_transition(
                state=latent_z,
                action=action,
                reward=result.reward,
                value=action_info["value"],
                log_prob=action_info["log_prob"],
                done=result.done,
                discrete_action=action_info.get("discrete_action_idx", 0),
            )

            next_frame_tensor = (
                torch.FloatTensor(result.observation).permute(2, 0, 1).unsqueeze(0) / 255.0
            )
            with torch.no_grad():
                next_perception = self.perception.process_frame(next_frame_tensor)
            next_z = next_perception["latent"].cpu().numpy().squeeze(0)
            self.replay_buffer.add(latent_z, action, result.reward, next_z, result.done)

        self._check_event_interactions()

        # Update ecology (day/night, weather, NPCs, resource collection)
        dt = 1.0 / 60.0 * self.speed_multiplier
        agent_pos = self.env.agent.position.tolist()
        ecology_update = self.ecology.update(
            dt=dt, agent_position=agent_pos, current_time=time.time(),
        )

        # Apply ecology reward modifiers
        eco_modifier = self.ecology.get_reward_modifier()
        for eco_event in ecology_update.get("events", []):
            if eco_event["type"] == "resource_collected":
                result = StepResult(
                    observation=result.observation,
                    reward=result.reward + eco_event["value"] * eco_modifier,
                    done=result.done,
                    info=result.info,
                )

        self.step_count += 1

        attention_np = perception["attention_map"].cpu().numpy()
        features_np = perception["features_downsampled"].cpu().numpy().squeeze(0)

        # Collect orb animation states for rendering
        orb_animations = []
        now = time.time()
        for event in self.event_injector.get_active_events():
            orb_animations.append(get_orb_animation_state(event, now))

        return {
            "frame": result.observation,
            "attention_map": attention_np,
            "features": features_np,
            "action_info": action_info,
            "reward": result.reward,
            "done": result.done,
            "state": result.info,
            "step": self.step_count,
            "ecology": ecology_update,
            "orb_animations": orb_animations,
        }

    def train_step(self) -> dict[str, Any] | None:
        """Run PPO update if buffer is full."""
        if not self.trainer.buffer.full:
            return None

        if self.current_latent is not None:
            z_tensor = torch.FloatTensor(self.current_latent).unsqueeze(0)
            with torch.no_grad():
                _, value = self.trainer.policy(z_tensor.to(self.trainer.device))
            last_value = value.item()
        else:
            last_value = 0.0

        stats = self.trainer.update(last_value)
        return stats

    def inject_question(self, text: str, user_id: str = "anon") -> dict[str, Any]:
        """Inject a question as a world event with multilingual support."""
        # Detect language and translate to English for processing
        ml_input = self.multilingual.process_input(text)
        english_text = ml_input["english_text"]
        source_lang = ml_input["source_lang"]

        event = self.event_injector.create_question_event(english_text)

        body_id = self.env.inject_entity(
            "query_orb",
            metadata={"event_id": event.event_id, "text": text},
        )
        event.body_id = body_id
        event.position = list(
            self.env.agent.position + np.array([
                np.cos(self.env.agent.orientation) * 2,
                np.sin(self.env.agent.orientation) * 2,
                0,
            ])
        )

        if self.current_latent is not None:
            self.memory.store(
                embedding=self.current_latent,
                event_type="question",
                context=english_text,
                metadata={"user_id": user_id, "event_id": event.event_id},
            )

        # Record interaction for retention
        user_store = self.retention.get_store(user_id)
        user_store.record_interaction(
            event_type="question",
            context=english_text,
            embedding=self.current_latent,
            metadata={
                "event_id": event.event_id,
                "original_text": text,
                "source_lang": source_lang,
            },
        )

        return {
            "event_id": event.event_id,
            "body_id": body_id,
            "position": event.position,
            "status": "injected",
            "source_lang": source_lang,
            "original_text": text,
            "english_text": english_text,
            "was_translated": ml_input.get("was_translated", False),
        }

    def process_question_response(self, event_id: str) -> dict[str, Any] | None:
        """Generate a response for a question event."""
        event = self.event_injector.active_events.get(event_id)
        if not event:
            return None

        if self.current_latent is not None:
            memories = self.memory.retrieve(self.current_latent, top_k=5)
        else:
            memories = []

        confidence = min(0.95, len(memories) * 0.15) if memories else 0.0

        raw_response = self.response_generator.generate(
            question=event.text,
            context_memories=memories,
            agent_state=self.env.get_state(),
            confidence=confidence,
        )

        ethical_response = format_ethical_response(
            answer=raw_response["text"],
            confidence=confidence,
            uncertainty_factors=["limited sensory context", "finite memory window"],
        )

        self.event_injector.resolve_event(event_id, ethical_response["text"], confidence)

        if self.current_latent is not None:
            self.memory.store(
                embedding=self.current_latent,
                event_type="response",
                context=event.text,
                response=ethical_response["text"],
                metadata={"confidence": confidence},
            )

        return ethical_response

    def get_full_state(self) -> dict[str, Any]:
        """Get complete simulation state for streaming."""
        env_state = self.env.get_state()
        ecology_state = self.ecology.get_state()
        user_store = self.retention.get_store(self.user_id)

        return {
            "environment": env_state,
            "training": {
                "total_steps": self.trainer.total_steps,
                "episode": self.episode_count,
                "stats": self.trainer.training_stats[-1]
                if self.trainer.training_stats else {},
            },
            "memory": {
                "episodic_size": self.memory.size,
                "replay_size": self.replay_buffer.size,
            },
            "events": {
                "active": len(self.event_injector.get_active_events()),
                "resolved": len(self.event_injector.get_resolved_events()),
            },
            "action": self.current_action_info,
            "speed": self.speed_multiplier,
            "paused": self.paused,
            "ecology": ecology_state,
            "procedural": {
                "seed": self.world_seed.seed,
                "difficulty": self.ecology.curriculum.difficulty,
                "heightmap_resolution": (
                    self.procedural_config.terrain.resolution
                ),
            },
            "retention": user_store.get_state(),
        }

    async def run_loop(self, target_fps: int = 60,
                       perception_fps: int = 15) -> None:
        """Main async simulation loop."""
        self.running = True
        frame = self.reset()
        state_interval = 1.0 / target_fps
        perception_interval = 1.0 / perception_fps
        last_perception_time = 0.0

        while self.running:
            if self.paused:
                await asyncio.sleep(0.1)
                continue

            start = time.time()
            step_result = self.step(frame)
            frame = step_result["frame"]

            state = self.get_full_state()
            await self.ws_manager.broadcast_state(state)

            now = time.time()
            if now - last_perception_time >= perception_interval:
                perception_payload = prepare_perception_payload(
                    frame=step_result["frame"],
                    attention_map=step_result["attention_map"],
                    action_info=step_result["action_info"],
                    features=step_result["features"],
                )
                await self.ws_manager.broadcast_perception(perception_payload)
                last_perception_time = now

            train_stats = self.train_step()
            if train_stats:
                await self.ws_manager.broadcast_state({
                    "type": "training_update",
                    "data": train_stats,
                })

            if step_result["done"]:
                frame = self.reset()

            elapsed = time.time() - start
            sleep_time = max(0, state_interval / self.speed_multiplier - elapsed)
            await asyncio.sleep(sleep_time)

    def stop(self) -> None:
        self.running = False

    def set_speed(self, multiplier: float) -> None:
        self.speed_multiplier = max(0.1, min(100.0, multiplier))

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def _check_event_interactions(self) -> None:
        """Check if agent is near any active events and trigger responses."""
        for event in self.event_injector.get_active_events():
            if event.body_id < 0:
                continue
            event_pos = np.array(event.position)
            dist = np.linalg.norm(self.env.agent.position[:2] - event_pos[:2])
            if dist < 1.5:
                self.process_question_response(event.event_id)
