"""Tests for PPO training."""

import numpy as np
import torch

from backend.learning.ppo import (
    DISCRETE_ACTIONS,
    ActorCritic,
    PPOConfig,
    PPOTrainer,
    RolloutBuffer,
)


def test_actor_critic():
    model = ActorCritic(latent_dim=256, action_dim=3, hidden_dim=256, discrete_action_dim=4)
    z = torch.randn(4, 256)
    dist, value = model(z)
    assert value.shape == (4, 1)
    assert dist.loc.shape == (4, 3)


def test_get_action():
    model = ActorCritic(latent_dim=256, action_dim=3)
    z = torch.randn(1, 256)
    action, log_prob, value, entropy = model.get_action(z)
    assert action.shape == (1, 3)
    assert log_prob.shape == (1,)
    assert value.shape == (1,)
    assert entropy.shape == (1,)


def test_rollout_buffer():
    buf = RolloutBuffer(buffer_size=100, latent_dim=256, action_dim=3)
    for _ in range(100):
        buf.add(
            state=np.random.randn(256).astype(np.float32),
            action=np.random.randn(3).astype(np.float32),
            reward=np.random.randn(),
            value=np.random.randn(),
            log_prob=np.random.randn(),
            done=False,
        )
    assert buf.full
    buf.compute_gae(last_value=0.0)
    batches = list(buf.get_batches(32))
    assert len(batches) > 0


def test_actor_critic_full_action():
    model = ActorCritic(latent_dim=64, action_dim=3, discrete_action_dim=4)
    z = torch.randn(1, 64)
    cont_action, cont_lp, disc_action, disc_lp, value, entropy = model.get_full_action(z)
    assert cont_action.shape == (1, 3)
    assert disc_action.shape == (1,)
    assert 0 <= disc_action.item() < 4


def test_ppo_trainer_select_action():
    config = PPOConfig(latent_dim=64, action_dim=3, hidden_dim=64, use_curiosity=False)
    trainer = PPOTrainer(config=config)
    z = np.random.randn(64).astype(np.float32)
    result = trainer.select_action(z)
    assert "action" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1
    assert "discrete_action" in result
    assert result["discrete_action"] in DISCRETE_ACTIONS
    assert "discrete_probs" in result
    assert len(result["discrete_probs"]) == 4


def test_ppo_trainer_update():
    config = PPOConfig(
        latent_dim=64, action_dim=3, hidden_dim=64,
        buffer_size=64, batch_size=16, ppo_epochs=2,
        use_curiosity=False,
    )
    trainer = PPOTrainer(config=config)

    for _ in range(64):
        z = np.random.randn(64).astype(np.float32)
        result = trainer.select_action(z)
        trainer.store_transition(
            state=z, action=result["action"],
            reward=np.random.randn(), value=result["value"],
            log_prob=result["log_prob"], done=False,
        )

    stats = trainer.update(last_value=0.0)
    assert "policy_loss" in stats
    assert "value_loss" in stats


def test_ppo_with_curiosity():
    config = PPOConfig(
        latent_dim=64, action_dim=3, hidden_dim=64,
        use_curiosity=True, curiosity_scale=0.1,
    )
    trainer = PPOTrainer(config=config)
    assert trainer.curiosity is not None

    z = np.random.randn(64).astype(np.float32)
    result = trainer.select_action(z)
    trainer.store_transition(
        state=z, action=result["action"],
        reward=0.0, value=result["value"],
        log_prob=result["log_prob"], done=False,
    )

    # Train curiosity
    states = np.random.randn(16, 64).astype(np.float32)
    loss = trainer.train_curiosity(states)
    assert loss is not None and loss >= 0.0
