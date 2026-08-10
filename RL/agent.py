from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from RL.config import AgentConfig, NetworkConfig
from RL.network import QNetwork
from RL.replay_buffer import ReplayBuffer, TransitionBatch


class DQNAgent:
    def __init__(
        self,
        agent_config: AgentConfig,
        network_config: NetworkConfig,
        replay_buffer: ReplayBuffer,
    ) -> None:
        self.config = agent_config
        self.replay_buffer = replay_buffer
        self.online_network = QNetwork(network_config)
        self.target_network = QNetwork(network_config)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=self.config.learning_rate
        )
        self._cache_capacity = network_config.cache_capacity
        self._env_steps = 0
        self._train_steps = 0
        self.last_batch_stats: dict[str, float] = {}

    @property
    def epsilon(self) -> float:
        decay_progress = min(1.0, self._env_steps / self.config.epsilon_decay_steps)
        return (
            self.config.epsilon_start
            + decay_progress * (self.config.epsilon_end - self.config.epsilon_start)
        )

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy:
            epsilon = self.epsilon
            self._env_steps += 1
            if random.random() < epsilon:
                return random.randrange(self._cache_capacity)

        self.online_network.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            q_values = self.online_network(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        self.online_network.train()
        return action

    def train_step(self) -> float:
        batch = self.replay_buffer.sample()
        states = torch.from_numpy(batch.states).float()
        actions = torch.from_numpy(batch.actions).long().unsqueeze(1)
        q_values = self.online_network(states).gather(1, actions).squeeze(1)
        targets = self.compute_td_targets(batch)
        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.config.grad_clip_norm
        )
        self.optimizer.step()
        self._train_steps += 1
        if self._train_steps % self.config.target_update_interval == 0:
            self.update_target_network()
        with torch.no_grad():
            self.last_batch_stats = {
                "q_mean": q_values.mean().item(),
                "q_std": q_values.std().item(),
                "q_max": q_values.max().item(),
                "target_mean": targets.mean().item(),
                "target_std": targets.std().item(),
                "target_max": targets.max().item(),
            }

        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())

    def compute_td_targets(self, batch: TransitionBatch) -> torch.Tensor:
        next_states = torch.from_numpy(batch.next_states).float()
        rewards = torch.from_numpy(batch.rewards).float()
        dones = torch.from_numpy(batch.dones).float()
        steps_elapsed = torch.from_numpy(batch.steps_elapsed).float()

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(dim=1).values
        discount = self.config.gamma ** steps_elapsed
        targets = rewards + discount * max_next_q * (1.0 - dones)
        return targets