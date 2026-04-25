"""Tests for the physics environment."""

import numpy as np
import pytest

from backend.simulation.physics_env import EnvConfig, PhysicsEnvironment


@pytest.fixture
def env():
    config = EnvConfig(render_width=64, render_height=64, max_steps=100)
    environment = PhysicsEnvironment(config=config)
    yield environment
    environment.close()


def test_env_reset(env):
    frame = env.reset()
    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.uint8


def test_env_step(env):
    env.reset()
    action = np.array([0.5, 0.0, 0.0])
    result = env.step(action)
    assert result.observation.shape == (64, 64, 3)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)
    assert "position" in result.info


def test_env_agent_moves(env):
    env.reset()
    initial_pos = env.agent.position.copy()
    for _ in range(10):
        env.step(np.array([1.0, 0.0, 0.0]))
    final_pos = env.agent.position
    assert not np.allclose(initial_pos, final_pos)


def test_env_collision(env):
    env.reset()
    assert len(env.objects) == env.config.num_cubes


def test_env_inject_entity(env):
    env.reset()
    body_id = env.inject_entity("query_orb")
    assert body_id >= 0
    assert len(env._injected_entities) == 1


def test_env_remove_entity(env):
    env.reset()
    body_id = env.inject_entity("query_orb")
    env.remove_entity(body_id)
    assert len(env._injected_entities) == 0


def test_env_get_state(env):
    env.reset()
    state = env.get_state()
    assert "agent" in state
    assert "objects" in state
    assert "position" in state["agent"]


def test_env_max_steps(env):
    env.reset()
    done = False
    for _ in range(200):
        result = env.step(np.array([0.0, 0.0, 0.0]))
        if result.done:
            done = True
            break
    assert done
