"""
PPO (Proximal Policy Optimization) implementation for the embodied agent.
State: latent z from vision encoder
Action: continuous [forward/backward, left/right, rotation]
      + discrete [interact, pick, drop]
Supports curiosity module (RND) for intrinsic motivation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Normal


@dataclass
class PPOConfig:
    latent_dim: int = 256
    action_dim: int = 3  # continuous actions
    discrete_action_dim: int = 4  # none, interact, pick, drop
    hidden_dim: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    buffer_size: int = 2048
    novelty_bonus: float = 0.1
    use_curiosity: bool = True
    curiosity_scale: float = 0.1


DISCRETE_ACTIONS = ["none", "interact", "pick", "drop"]


class ActorCritic(nn.Module):
    """Combined actor-critic network for PPO with continuous + discrete actions."""

    def __init__(self, latent_dim: int = 256, action_dim: int = 3,
                 hidden_dim: int = 256, discrete_action_dim: int = 4):
        super().__init__()
        self.discrete_action_dim = discrete_action_dim

        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        self.discrete_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, discrete_action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: Tensor) -> tuple[Normal, Tensor]:
        shared_features = self.shared(z)
        action_mean = self.actor_mean(shared_features)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        value = self.critic(shared_features)
        return dist, value

    def forward_full(self, z: Tensor) -> tuple[Normal, Categorical, Tensor]:
        """Forward pass returning both continuous and discrete distributions."""
        shared_features = self.shared(z)
        action_mean = self.actor_mean(shared_features)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        cont_dist = Normal(action_mean, action_std)
        discrete_logits = self.discrete_head(shared_features)
        disc_dist = Categorical(logits=discrete_logits)
        value = self.critic(shared_features)
        return cont_dist, disc_dist, value

    def get_action(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample action, return (action, log_prob, value, entropy)."""
        dist, value = self(z)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, value.squeeze(-1), entropy

    def get_full_action(
        self, z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample both continuous and discrete actions."""
        cont_dist, disc_dist, value = self.forward_full(z)
        cont_action = torch.clamp(cont_dist.sample(), -1.0, 1.0)
        cont_log_prob = cont_dist.log_prob(cont_action).sum(dim=-1)
        cont_entropy = cont_dist.entropy().sum(dim=-1)
        disc_action = disc_dist.sample()
        disc_log_prob = disc_dist.log_prob(disc_action)
        disc_entropy = disc_dist.entropy()
        return (
            cont_action, cont_log_prob,
            disc_action, disc_log_prob,
            value.squeeze(-1),
            cont_entropy + disc_entropy,
        )

    def evaluate_action(self, z: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate a previously taken continuous action (legacy, continuous only)."""
        dist, value = self(z)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value.squeeze(-1), entropy

    def evaluate_full_action(
        self, z: Tensor, cont_action: Tensor, disc_action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate both continuous and discrete actions for PPO update."""
        cont_dist, disc_dist, value = self.forward_full(z)
        cont_log_prob = cont_dist.log_prob(cont_action).sum(dim=-1)
        disc_log_prob = disc_dist.log_prob(disc_action)
        total_log_prob = cont_log_prob + disc_log_prob
        total_entropy = cont_dist.entropy().sum(dim=-1) + disc_dist.entropy()
        return total_log_prob, value.squeeze(-1), total_entropy


class RolloutBuffer:
    """Buffer to store rollout transitions for PPO updates."""

    def __init__(self, buffer_size: int, latent_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> None:
        self.states = np.zeros((self.buffer_size, self.latent_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.discrete_actions = np.zeros(self.buffer_size, dtype=np.int64)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        self.ptr = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            value: float, log_prob: float, done: bool,
            discrete_action: int = 0) -> None:
        if self.ptr >= self.buffer_size:
            return
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.discrete_actions[self.ptr] = discrete_action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)
        self.ptr += 1

    @property
    def full(self) -> bool:
        return self.ptr >= self.buffer_size

    def compute_gae(self, last_value: float, gamma: float = 0.99,
                    lam: float = 0.95) -> None:
        """Compute Generalized Advantage Estimation."""
        gae = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_batches(self, batch_size: int):
        """Yield random mini-batches."""
        indices = np.random.permutation(self.ptr)
        for start in range(0, self.ptr, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield {
                "states": torch.FloatTensor(self.states[batch_idx]),
                "actions": torch.FloatTensor(self.actions[batch_idx]),
                "discrete_actions": torch.LongTensor(self.discrete_actions[batch_idx]),
                "old_log_probs": torch.FloatTensor(self.log_probs[batch_idx]),
                "advantages": torch.FloatTensor(self.advantages[batch_idx]),
                "returns": torch.FloatTensor(self.returns[batch_idx]),
            }


class PPOTrainer:
    """PPO training loop manager."""

    def __init__(self, config: PPOConfig | None = None, device: str = "cpu"):
        self.config = config or PPOConfig()
        self.device = torch.device(device)

        self.policy = ActorCritic(
            latent_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            discrete_action_dim=self.config.discrete_action_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.lr
        )

        self.buffer = RolloutBuffer(
            buffer_size=self.config.buffer_size,
            latent_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
        )

        self.curiosity: "RNDModule | None" = None
        if self.config.use_curiosity:
            from backend.learning.curiosity import RNDConfig, RNDModule
            rnd_config = RNDConfig(
                input_dim=self.config.latent_dim,
                reward_scale=self.config.curiosity_scale,
            )
            self.curiosity = RNDModule(config=rnd_config, device=device)

        self.training_stats: list[dict] = []
        self._state_embeddings: list[np.ndarray] = []
        self.total_steps = 0

    def select_action(self, latent_z: np.ndarray) -> dict:
        """Select both continuous and discrete actions given latent state."""
        z_tensor = torch.FloatTensor(latent_z).unsqueeze(0).to(self.device)

        with torch.no_grad():
            (
                cont_action, cont_log_prob,
                disc_action, disc_log_prob,
                value, entropy,
            ) = self.policy.get_full_action(z_tensor)
            cont_dist, disc_dist, _ = self.policy.forward_full(z_tensor)

        action_np = cont_action.cpu().numpy().squeeze(0)
        disc_idx = disc_action.item()
        confidence = 1.0 - entropy.item() / (self.config.action_dim * 2.0 + 2.0)
        confidence = max(0.0, min(1.0, confidence))

        return {
            "action": action_np.tolist(),
            "discrete_action": DISCRETE_ACTIONS[disc_idx],
            "discrete_action_idx": disc_idx,
            "discrete_probs": disc_dist.probs.cpu().numpy().squeeze(0).tolist(),
            "log_prob": cont_log_prob.item() + disc_log_prob.item(),
            "value": value.item(),
            "entropy": entropy.item(),
            "confidence": confidence,
            "action_mean": cont_dist.loc.cpu().numpy().squeeze(0).tolist(),
            "action_std": cont_dist.scale.cpu().numpy().squeeze(0).tolist(),
        }

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                         reward: float, value: float, log_prob: float,
                         done: bool, discrete_action: int = 0) -> None:
        novelty = self._compute_novelty(state)
        reward += self.config.novelty_bonus * novelty

        if self.curiosity is not None:
            intrinsic_reward = self.curiosity.compute_intrinsic_reward(state)
            reward += intrinsic_reward

        self.buffer.add(state, action, reward, value, log_prob, done,
                        discrete_action=discrete_action)
        self._state_embeddings.append(state.copy())
        if len(self._state_embeddings) > 10000:
            self._state_embeddings = self._state_embeddings[-5000:]
        self.total_steps += 1

    def update(self, last_value: float) -> dict:
        """Run PPO update on collected buffer."""
        self.buffer.compute_gae(last_value, self.config.gamma, self.config.gae_lambda)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.config.ppo_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                states = batch["states"].to(self.device)
                actions = batch["actions"].to(self.device)
                discrete_actions = batch["discrete_actions"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                new_log_probs, values, entropy = self.policy.evaluate_full_action(
                    states, actions, discrete_actions
                )

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon,
                    )
                    * advantages
                )

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coeff * value_loss
                    + self.config.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        stats = {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "total_steps": self.total_steps,
            "buffer_size": self.buffer.ptr,
        }
        self.training_stats.append(stats)
        self.buffer.reset()
        return stats

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "training_stats": self.training_stats,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        self.training_stats = checkpoint["training_stats"]

    def train_curiosity(self, states: np.ndarray) -> float | None:
        """Train the RND curiosity module on a batch of states."""
        if self.curiosity is None:
            return None
        return self.curiosity.train_step(states)

    def _compute_novelty(self, state: np.ndarray) -> float:
        """Novelty bonus based on distance from recent states."""
        if len(self._state_embeddings) < 10:
            return 1.0
        recent = np.array(self._state_embeddings[-100:])
        distances = np.linalg.norm(recent - state, axis=1)
        return float(np.mean(np.sort(distances)[:5]))
