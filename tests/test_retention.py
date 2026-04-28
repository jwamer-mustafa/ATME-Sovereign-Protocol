"""Tests for retention mechanics."""

import numpy as np

from backend.retention.user_memory import (
    DEFAULT_AGENTS,
    RetentionManager,
    UserMemoryStore,
)


def test_user_memory_store_creation():
    """User memory store initializes correctly."""
    store = UserMemoryStore(user_id="test_user")
    assert store.user_id == "test_user"
    assert len(store.interactions) == 0
    assert len(store.badges) == 0


def test_record_interaction():
    """Interactions are recorded correctly."""
    store = UserMemoryStore(user_id="test_user")
    interaction = store.record_interaction(
        event_type="question",
        context="What is this?",
        confidence=0.8,
    )
    assert interaction.user_id == "test_user"
    assert interaction.event_type == "question"
    assert len(store.interactions) == 1


def test_retrieve_similar():
    """Similar interactions are retrieved by embedding cosine similarity."""
    store = UserMemoryStore(user_id="test_user")

    emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    emb3 = np.array([0.9, 0.1, 0.0], dtype=np.float32)

    store.record_interaction("q", "first", embedding=emb1)
    store.record_interaction("q", "second", embedding=emb2)
    store.record_interaction("q", "third", embedding=emb3)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = store.retrieve_similar(query, top_k=2)
    assert len(results) == 2
    assert results[0].context == "first"
    assert results[1].context == "third"


def test_award_badge():
    """Badges are awarded and not duplicated."""
    store = UserMemoryStore(user_id="test_user")

    b1 = store.award_badge("Explorer", "Explored 10 areas", skill_level=1.0)
    assert b1.name == "Explorer"

    b2 = store.award_badge("Explorer", "Explored 10 areas", skill_level=2.0)
    assert b2.skill_level == 2.0  # Updated, not duplicated
    assert len(store.badges) == 1


def test_generate_challenge():
    """Challenges are generated with correct structure."""
    store = UserMemoryStore(user_id="test_user")
    challenge = store.generate_challenge(difficulty=0.5)
    assert "type" in challenge
    assert "description" in challenge
    assert "difficulty" in challenge
    assert "reward" in challenge
    assert "time_limit" in challenge
    assert challenge["difficulty"] == 0.5


def test_select_agent():
    """Agent profiles can be selected."""
    store = UserMemoryStore(user_id="test_user")
    assert store.selected_agent == "agent_explorer"

    profile = store.select_agent("agent_cautious")
    assert profile is not None
    assert profile.name == "Guardian"
    assert store.selected_agent == "agent_cautious"

    invalid = store.select_agent("nonexistent")
    assert invalid is None


def test_default_agents():
    """Default agents have correct structure."""
    assert len(DEFAULT_AGENTS) == 4
    for agent in DEFAULT_AGENTS:
        assert agent.agent_id.startswith("agent_")
        assert agent.name
        assert "curiosity" in agent.traits
        assert "caution" in agent.traits


def test_retention_manager():
    """Retention manager creates and retrieves user stores."""
    manager = RetentionManager()
    store1 = manager.get_store("user_1")
    store2 = manager.get_store("user_1")
    assert store1 is store2  # Same instance

    store3 = manager.get_store("user_2")
    assert store3 is not store1


def test_get_state():
    """State serialization produces correct structure."""
    store = UserMemoryStore(user_id="test_user")
    store.record_interaction("test", "hello")
    store.award_badge("First Steps", "Complete the tutorial")
    state = store.get_state()

    assert state["user_id"] == "test_user"
    assert state["interaction_count"] == 1
    assert len(state["badges"]) == 1
    assert len(state["available_agents"]) == 4
    assert "evolution" in state


def test_evolution_comparison_no_history():
    """Evolution comparison handles no history gracefully."""
    store = UserMemoryStore(user_id="test_user")
    comparison = store.get_evolution_comparison()
    assert not comparison["has_history"]
