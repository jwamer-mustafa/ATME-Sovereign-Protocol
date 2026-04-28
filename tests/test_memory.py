"""Tests for episodic memory and replay buffer."""

import numpy as np

from backend.memory.episodic import EpisodicMemory, ReplayBuffer


def test_episodic_store_retrieve():
    mem = EpisodicMemory(capacity=100)
    embedding = np.random.randn(256).astype(np.float32)
    mem.store(embedding, event_type="question", context="What is this?")
    assert mem.size == 1

    results = mem.retrieve(embedding, top_k=1)
    assert len(results) == 1
    assert results[0]["context"] == "What is this?"
    assert results[0]["similarity"] > 0.9


def test_episodic_cosine_similarity():
    mem = EpisodicMemory(capacity=100)
    e1 = np.array([1.0, 0.0, 0.0] + [0.0] * 253, dtype=np.float32)
    e2 = np.array([0.0, 1.0, 0.0] + [0.0] * 253, dtype=np.float32)
    e3 = np.array([0.9, 0.1, 0.0] + [0.0] * 253, dtype=np.float32)

    mem.store(e1, "event", "similar to query")
    mem.store(e2, "event", "different from query")

    results = mem.retrieve(e3, top_k=2)
    assert results[0]["context"] == "similar to query"


def test_episodic_capacity():
    mem = EpisodicMemory(capacity=10)
    for i in range(20):
        mem.store(np.random.randn(256), "event", f"event_{i}")
    assert mem.size == 10


def test_replay_buffer():
    buf = ReplayBuffer(capacity=100, state_dim=64, action_dim=3)
    for _ in range(50):
        buf.add(
            state=np.random.randn(64).astype(np.float32),
            action=np.random.randn(3).astype(np.float32),
            reward=1.0,
            next_state=np.random.randn(64).astype(np.float32),
            done=False,
        )
    assert buf.size == 50

    batch = buf.sample(16)
    assert batch["states"].shape == (16, 64)
    assert batch["actions"].shape == (16, 3)
