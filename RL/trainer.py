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
        self._num_eval_episodes = 5

    def warmup(self) -> None:
        min_required = max(self.config.warmup_steps, self.replay_buffer.config.batch_size)

        self.environment.reset()
        while len(self.replay_buffer) < min_required:
            if self.environment.done:
                self.environment.reset()

            state = self.environment.current_state()
            action = self.agent.select_action(state)
            transition = self._collect_transition(action)
            self.replay_buffer.push(transition)

    def train(self) -> None:
        self.warmup()
        self.environment.reset()

        for step in range(1, self.config.num_training_steps + 1):
            if self.environment.done:
                self.environment.reset()

            state = self.environment.current_state()
            action = self.agent.select_action(state)
            transition = self._collect_transition(action)
            self.replay_buffer.push(transition)

            self._last_loss = self.agent.train_step()

            if step % self.config.log_interval == 0:
                self.log(step)

            if step % self.config.eval_interval == 0:
                self.evaluate()

    def evaluate(self) -> EvaluationResult:
        saved_epsilon_start = self.agent.config.epsilon_start
        saved_epsilon_end = self.agent.config.epsilon_end
        self.agent.config.epsilon_start = 0.0
        self.agent.config.epsilon_end = 0.0

        total_hits = 0
        total_accesses = 0

        try:
            for _ in range(self._num_eval_episodes):
                self.environment.reset()
                last_info: dict = {"num_hits": 0, "num_misses": 0}

                while not self.environment.done:
                    state = self.environment.current_state()
                    action = self.agent.select_action(state)
                    result = self.environment.step(action)
                    last_info = result.info

                total_hits += last_info["num_hits"]
                total_accesses += last_info["num_hits"] + last_info["num_misses"]
        finally:
            self.agent.config.epsilon_start = saved_epsilon_start
            self.agent.config.epsilon_end = saved_epsilon_end

        hit_rate = total_hits / total_accesses if total_accesses > 0 else 0.0
        return EvaluationResult(hit_rate=hit_rate, num_episodes=self._num_eval_episodes)

    def log(self, step: int) -> None:
        loss_str = f"{self._last_loss:.4f}" if self._last_loss is not None else "n/a"
        print(
            f"step {step}/{self.config.num_training_steps} "
            f"loss={loss_str} "
            f"buffer_size={len(self.replay_buffer)}"
        )

    def _collect_transition(self, action: int) -> Transition:
        state = self.environment.current_state()
        result = self.environment.step(action)
        return Transition(
            state=state,
            action=action,
            reward=result.reward,
            next_state=result.next_state,
            done=result.done,
        )