
from dataclasses import dataclass

from RL.agent import DQNAgent
from RL.config import TrainerConfig
from RL.environment import CacheEvictionEnvironment
from RL.replay_buffer import ReplayBuffer, Transition


@dataclass
class EvaluationResult:
    hit_rate: float
    num_episodes: int


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        environment: CacheEvictionEnvironment,
        agent: DQNAgent,
        replay_buffer: ReplayBuffer,
    ) -> None:
        self.config = config
        self.environment = environment
        self.agent = agent
        self.replay_buffer = replay_buffer
        

    def warmup(self) -> None:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> EvaluationResult:
        raise NotImplementedError

    def log(self, step: int) -> None:
        raise NotImplementedError

    def _collect_transition(self, action: int) -> Transition:
        raise NotImplementedError