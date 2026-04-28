"""
Memory System:
1) Replay Buffer — for RL training (s, a, r, s')
2) Episodic Memory — for experience recall (embedding-based retrieval)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Episode:
    embedding: np.ndarray
    event_type: str
    context: str
    response: str | None = None
    reward: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """
    Stores episodic memories keyed by latent embeddings.
    Retrieval via cosine similarity for context-aware responses.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.episodes: list[Episode] = []

    def store(self, embedding: np.ndarray, event_type: str, context: str,
              response: str | None = None, reward: float = 0.0,
              metadata: dict[str, Any] | None = None) -> None:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        episode = Episode(
            embedding=embedding,
            event_type=event_type,
            context=context,
            response=response,
            reward=reward,
            metadata=metadata or {},
        )
        self.episodes.append(episode)

        if len(self.episodes) > self.capacity:
            self.episodes = self.episodes[-self.capacity:]

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5,
                 event_type: str | None = None) -> list[dict[str, Any]]:
        """Retrieve most similar memories using cosine similarity."""
        if not self.episodes:
            return []

        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        candidates = self.episodes
        if event_type:
            candidates = [e for e in candidates if e.event_type == event_type]

        if not candidates:
            return []

        embeddings = np.array([e.embedding for e in candidates])
        similarities = embeddings @ query_embedding

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            ep = candidates[idx]
            results.append({
                "similarity": float(similarities[idx]),
                "event_type": ep.event_type,
                "context": ep.context,
                "response": ep.response,
                "reward": ep.reward,
                "timestamp": ep.timestamp,
                "metadata": ep.metadata,
            })

        return results

    def get_recent(self, n: int = 10) -> list[dict[str, Any]]:
        recent = self.episodes[-n:]
        return [
            {
                "event_type": ep.event_type,
                "context": ep.context,
                "response": ep.response,
                "reward": ep.reward,
                "timestamp": ep.timestamp,
            }
            for ep in reversed(recent)
        ]

    @property
    def size(self) -> int:
        return len(self.episodes)

    def clear(self) -> None:
        self.episodes.clear()


class ReplayBuffer:
    """Standard replay buffer for RL training transitions (s, a, r, s', done)."""

    def __init__(self, capacity: int = 100000, state_dim: int = 256,
                 action_dim: int = 3):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        idx = self.ptr % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }
