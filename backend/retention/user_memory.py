"""
Retention mechanics: user-bound memory, evolution tracking, personalized challenges,
multi-agent profiles, and explicit uncertainty.

Goal: "Feels like a mind that evolves and remembers you" — without consciousness claims.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class UserInteraction:
    """A single user interaction record."""
    interaction_id: str
    user_id: str
    timestamp: float
    event_type: str
    context: str
    embedding: np.ndarray | None = None
    response: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillBadge:
    """A badge representing an acquired agent capability."""
    badge_id: str
    name: str
    description: str
    acquired_at: float
    skill_level: float = 0.0
    category: str = "general"


@dataclass
class AgentProfile:
    """A distinct agent personality/behavior profile."""
    agent_id: str
    name: str
    behavior_style: str  # "cautious", "explorer", "social", "analytical"
    traits: dict[str, float] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self):
        default_traits = {
            "curiosity": 0.5,
            "caution": 0.5,
            "social": 0.5,
            "persistence": 0.5,
        }
        for k, v in default_traits.items():
            self.traits.setdefault(k, v)


# Pre-defined agent profiles for multi-agent selection
DEFAULT_AGENTS: list[AgentProfile] = [
    AgentProfile(
        agent_id="agent_explorer",
        name="Explorer",
        behavior_style="explorer",
        traits={"curiosity": 0.9, "caution": 0.2, "social": 0.5, "persistence": 0.8},
        description="Driven by curiosity, explores aggressively, takes risks.",
    ),
    AgentProfile(
        agent_id="agent_cautious",
        name="Guardian",
        behavior_style="cautious",
        traits={"curiosity": 0.3, "caution": 0.9, "social": 0.4, "persistence": 0.7},
        description="Careful and methodical, avoids unnecessary risks.",
    ),
    AgentProfile(
        agent_id="agent_social",
        name="Companion",
        behavior_style="social",
        traits={"curiosity": 0.5, "caution": 0.4, "social": 0.9, "persistence": 0.5},
        description="Responsive to user input, prioritizes interaction.",
    ),
    AgentProfile(
        agent_id="agent_analyst",
        name="Analyst",
        behavior_style="analytical",
        traits={"curiosity": 0.6, "caution": 0.6, "social": 0.3, "persistence": 0.9},
        description="Data-driven, focuses on optimal strategies.",
    ),
]


@dataclass
class EvolutionSnapshot:
    """A snapshot of agent skill levels at a point in time."""
    timestamp: float
    total_reward: float
    episode_count: int
    skills: dict[str, float]
    badges: list[str]


class UserMemoryStore:
    """
    Per-user memory storage for retention.
    Stores interactions, tracks evolution, generates personalized challenges.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.interactions: list[UserInteraction] = []
        self.badges: dict[str, SkillBadge] = {}
        self.evolution_timeline: list[EvolutionSnapshot] = []
        self.selected_agent: str = "agent_explorer"
        self._interaction_counter = 0
        self._session_start = time.time()
        self._total_session_time = 0.0

    def record_interaction(
        self,
        event_type: str,
        context: str,
        embedding: np.ndarray | None = None,
        response: str = "",
        confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> UserInteraction:
        """Record a user interaction."""
        self._interaction_counter += 1
        interaction = UserInteraction(
            interaction_id=f"int_{self.user_id}_{self._interaction_counter}",
            user_id=self.user_id,
            timestamp=time.time(),
            event_type=event_type,
            context=context,
            embedding=embedding,
            response=response,
            confidence=confidence,
            metadata=metadata or {},
        )
        self.interactions.append(interaction)
        return interaction

    def retrieve_similar(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[UserInteraction]:
        """Retrieve most similar past interactions by embedding cosine similarity."""
        scored: list[tuple[float, UserInteraction]] = []

        for interaction in self.interactions:
            if interaction.embedding is None:
                continue
            sim = float(np.dot(embedding, interaction.embedding) / (
                max(np.linalg.norm(embedding), 1e-8)
                * max(np.linalg.norm(interaction.embedding), 1e-8)
            ))
            scored.append((sim, interaction))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def award_badge(
        self,
        name: str,
        description: str,
        category: str = "general",
        skill_level: float = 1.0,
    ) -> SkillBadge:
        """Award a skill badge to the user's agent."""
        badge_id = f"badge_{name.lower().replace(' ', '_')}"

        if badge_id in self.badges:
            existing = self.badges[badge_id]
            existing.skill_level = max(existing.skill_level, skill_level)
            return existing

        badge = SkillBadge(
            badge_id=badge_id,
            name=name,
            description=description,
            acquired_at=time.time(),
            skill_level=skill_level,
            category=category,
        )
        self.badges[badge_id] = badge
        return badge

    def record_evolution(
        self,
        total_reward: float,
        episode_count: int,
        skills: dict[str, float] | None = None,
    ) -> EvolutionSnapshot:
        """Record a snapshot of agent evolution."""
        snapshot = EvolutionSnapshot(
            timestamp=time.time(),
            total_reward=total_reward,
            episode_count=episode_count,
            skills=skills or {},
            badges=list(self.badges.keys()),
        )
        self.evolution_timeline.append(snapshot)
        return snapshot

    def generate_challenge(self, difficulty: float = 0.5) -> dict[str, Any]:
        """Generate a personalized challenge based on user history."""
        challenge_types = [
            {
                "type": "reach_target",
                "description": "Navigate to the highlighted target zone",
                "base_reward": 2.0,
            },
            {
                "type": "collect_resources",
                "description": "Collect all energy resources in the area",
                "base_reward": 3.0,
            },
            {
                "type": "survive_storm",
                "description": "Survive a storm event without losing energy",
                "base_reward": 4.0,
            },
            {
                "type": "explore_area",
                "description": "Explore unvisited regions of the environment",
                "base_reward": 2.5,
            },
            {
                "type": "interact_npcs",
                "description": "Approach and interact with wandering NPCs",
                "base_reward": 3.5,
            },
        ]

        seen_types = {
            i.metadata.get("challenge_type")
            for i in self.interactions
            if i.event_type == "challenge"
        }

        unseen = [c for c in challenge_types if c["type"] not in seen_types]
        choices = unseen if unseen else challenge_types
        rng = np.random.default_rng()
        challenge = dict(choices[rng.integers(0, len(choices))])

        challenge["difficulty"] = difficulty
        challenge["reward"] = challenge["base_reward"] * (1.0 + difficulty)
        challenge["time_limit"] = max(30, int(120 * (1 - difficulty * 0.5)))

        return challenge

    def get_evolution_comparison(self) -> dict[str, Any]:
        """Get comparison between current and previous performance."""
        if len(self.evolution_timeline) < 2:
            return {
                "has_history": False,
                "message": "Insufficient data for comparison",
            }

        latest = self.evolution_timeline[-1]
        earliest = self.evolution_timeline[0]

        one_day_ago = time.time() - 86400
        daily_snapshots = [s for s in self.evolution_timeline if s.timestamp >= one_day_ago]
        daily_baseline = daily_snapshots[0] if daily_snapshots else earliest

        return {
            "has_history": True,
            "current": {
                "reward": latest.total_reward,
                "episodes": latest.episode_count,
                "badges": len(latest.badges),
            },
            "daily_change": {
                "reward_delta": latest.total_reward - daily_baseline.total_reward,
                "episode_delta": latest.episode_count - daily_baseline.episode_count,
                "new_badges": len(latest.badges) - len(daily_baseline.badges),
            },
            "all_time_change": {
                "reward_delta": latest.total_reward - earliest.total_reward,
                "episode_delta": latest.episode_count - earliest.episode_count,
                "total_badges": len(latest.badges),
            },
        }

    def select_agent(self, agent_id: str) -> AgentProfile | None:
        """Select a specific agent profile."""
        for profile in DEFAULT_AGENTS:
            if profile.agent_id == agent_id:
                self.selected_agent = agent_id
                return profile
        return None

    def get_selected_agent(self) -> AgentProfile:
        """Get the currently selected agent profile."""
        for profile in DEFAULT_AGENTS:
            if profile.agent_id == self.selected_agent:
                return profile
        return DEFAULT_AGENTS[0]

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for streaming/API."""
        return {
            "user_id": self.user_id,
            "interaction_count": len(self.interactions),
            "badges": [
                {
                    "id": b.badge_id,
                    "name": b.name,
                    "description": b.description,
                    "skill_level": b.skill_level,
                    "category": b.category,
                }
                for b in self.badges.values()
            ],
            "selected_agent": self.selected_agent,
            "available_agents": [
                {
                    "id": a.agent_id,
                    "name": a.name,
                    "style": a.behavior_style,
                    "description": a.description,
                    "traits": a.traits,
                }
                for a in DEFAULT_AGENTS
            ],
            "evolution": self.get_evolution_comparison(),
        }


class RetentionManager:
    """Manages per-user memory stores."""

    def __init__(self):
        self._stores: dict[str, UserMemoryStore] = {}

    def get_store(self, user_id: str) -> UserMemoryStore:
        """Get or create a user memory store."""
        if user_id not in self._stores:
            self._stores[user_id] = UserMemoryStore(user_id)
        return self._stores[user_id]

    def get_all_user_ids(self) -> list[str]:
        return list(self._stores.keys())
