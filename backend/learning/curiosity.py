"""
Curiosity-driven exploration using Random Network Distillation (RND).
The predictor network tries to match the output of a fixed random target network.
High prediction error = novel state = intrinsic reward.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class RNDConfig:
    input_dim: int = 256
    hidden_dim: int = 128
    output_dim: int = 64
    lr: float = 1e-3
    reward_scale: float = 0.1
    running_mean_decay: float = 0.99
    max_reward: float = 5.0


class RNDTargetNetwork(nn.Module):
    """Fixed random target network (never trained)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDPredictorNetwork(nn.Module):
    """Predictor network trained to match the target (trainable)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDModule:
    """
    Random Network Distillation for intrinsic motivation.
    Prediction error on novel states drives exploration.
    """

    def __init__(self, config: RNDConfig | None = None, device: str = "cpu"):
        self.config = config or RNDConfig()
        self.device = device

        self.target = RNDTargetNetwork(
            self.config.input_dim,
            self.config.hidden_dim,
            self.config.output_dim,
        ).to(device)

        self.predictor = RNDPredictorNetwork(
            self.config.input_dim,
            self.config.hidden_dim,
            self.config.output_dim,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=self.config.lr,
        )

        self._running_mean = 0.0
        self._running_var = 1.0
        self._count = 0
        self._recent_errors: deque[float] = deque(maxlen=1000)

    @torch.no_grad()
    def compute_intrinsic_reward(self, state: np.ndarray) -> float:
        """
        Compute intrinsic reward for a state.
        High prediction error = novelty = high intrinsic reward.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        target_features = self.target(state_t)
        predicted_features = self.predictor(state_t)

        error = ((target_features - predicted_features) ** 2).mean().item()

        self._count += 1
        self._running_mean = (
            self.config.running_mean_decay * self._running_mean
            + (1 - self.config.running_mean_decay) * error
        )
        self._running_var = (
            self.config.running_mean_decay * self._running_var
            + (1 - self.config.running_mean_decay) * (error - self._running_mean) ** 2
        )

        std = max(1e-8, self._running_var ** 0.5)
        normalized_error = (error - self._running_mean) / std

        intrinsic_reward = float(np.clip(
            normalized_error * self.config.reward_scale,
            0.0,
            self.config.max_reward,
        ))

        self._recent_errors.append(error)
        return intrinsic_reward

    def train_step(self, states: np.ndarray) -> float:
        """
        Train the predictor to match the target on a batch of states.
        Returns the training loss.
        """
        states_t = torch.FloatTensor(states).to(self.device)

        with torch.no_grad():
            target_features = self.target(states_t)

        predicted_features = self.predictor(states_t)
        loss = ((target_features - predicted_features) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @property
    def novelty_stats(self) -> dict[str, float]:
        """Get statistics about recent novelty estimates."""
        if not self._recent_errors:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}

        errors = list(self._recent_errors)
        return {
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),
            "max": float(np.max(errors)),
            "min": float(np.min(errors)),
        }
