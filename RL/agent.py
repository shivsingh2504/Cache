
import numpy as np
import torch

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
        
    def select_action(self, state: np.ndarray) -> int:
        raise NotImplementedError

    def train_step(self) -> float:
        raise NotImplementedError

    def update_target_network(self) -> None:
        raise NotImplementedError

    def compute_td_targets(self, batch: TransitionBatch) -> torch.Tensor:
        raise NotImplementedError