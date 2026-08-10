from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

from RL.agent import DQNAgent
from RL.config import TrainerConfig
from RL.environment import CacheEvictionEnvironment
from RL.replay_buffer import ReplayBuffer, Transition

_EVAL_PROGRESS_INTERVAL = 5_000


@dataclass
class EvaluationResult:
    hit_rate: float
    num_episodes: int
    num_hits: int = 0
    num_misses: int = 0


def _format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(max(seconds, 0))))


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        environment: CacheEvictionEnvironment,
        eval_environment: CacheEvictionEnvironment,
        agent: DQNAgent,
        replay_buffer: ReplayBuffer,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.config = config
        self.environment = environment
        self.eval_environment = eval_environment
        self.agent = agent
        self.replay_buffer = replay_buffer
        # Step 1 of the eval-noise investigation: was 1, which meant every
        # eval checkpoint reported hit rate from a single fixed-trace
        # realization. Bumped to 5 so each checkpoint reports an averaged
        # hit rate across 5 independent episodes, separating genuine
        # policy degradation from single-trace sampling noise. This is
        # the ONLY change in this file -- no DQN, replay, reward, state,
        # epsilon, workload, LSTM, or optimizer logic is touched.
        self._num_eval_episodes = 5
        self._last_loss: float | None = None
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._best_hit_rate = -1.0
        self.eval_history: list[dict[str, float]] = []

        self._start_time: float | None = None
        self._reward_sum_since_log = 0.0
        self._reward_count_since_log = 0

        # Stage 4 instrumentation: raw per-transition return and segment
        # length since the last diagnostic print, so distributional stats
        # (not just a running mean) can be reported. Cleared each time
        # log_diagnostics() runs (every eval_interval steps, alongside
        # evaluation) rather than every log_interval, to keep the sample
        # window matched to the eval cadence used in the audit.
        self._diag_rewards: list[float] = []
        self._diag_segment_lengths: list[int] = []

    def warmup(self) -> None:
        min_required = max(self.config.warmup_steps, self.replay_buffer.config.batch_size)

        print(f"[Warmup] filling replay buffer to {min_required:,} transitions...")
        self.environment.reset()
        cache_capacity = self.environment.config.cache_capacity
        while len(self.replay_buffer) < min_required:
            if self.environment.done:
                self.environment.reset()

            # Uniformly random action, not agent.select_action(). Warmup
            # transitions should never be policy-informed (the online
            # network is untrained anyway), and agent.select_action()'s
            # non-greedy path increments the epsilon-decay step counter as
            # a side effect -- routing warmup through it silently consumed
            # part of the epsilon_decay_steps budget before real training
            # even started (5,000 warmup steps = 10% of a 50,000-step
            # schedule), so epsilon began training already decayed below
            # its configured start value. Sampling the random action
            # directly here fills the buffer identically but leaves
            # agent._env_steps, and therefore epsilon, untouched until the
            # training loop actually begins.
            action = random.randrange(cache_capacity)
            transition = self._collect_transition(action)
            self.replay_buffer.push(transition)
        print(f"[Warmup] complete. buffer_size={len(self.replay_buffer):,}")

    def train(self) -> None:
        self._start_time = time.monotonic()
        self.warmup()
        self.environment.reset()

        for step in range(1, self.config.num_training_steps + 1):
            if self.environment.done:
                self.environment.reset()

            state = self.environment.current_state()
            action = self.agent.select_action(state)
            transition = self._collect_transition(action)
            self.replay_buffer.push(transition)

            self._reward_sum_since_log += transition.reward
            self._reward_count_since_log += 1
            self._diag_rewards.append(transition.reward)
            self._diag_segment_lengths.append(transition.steps_elapsed)

            self._last_loss = self.agent.train_step()

            if step % self.config.log_interval == 0:
                self.log(step)

            if step % self.config.eval_interval == 0:
                print(f"[Eval] running evaluation ({self._num_eval_episodes} episode(s))...")
                result = self.evaluate()
                print(
                    f"[Eval] step={step:,} "
                    f"hit_rate={result.hit_rate:.2%} "
                    f"epsilon={self.agent.epsilon:.3f}"
                )
                if result.hit_rate > self._best_hit_rate:
                    self._best_hit_rate = result.hit_rate
                    self.save_checkpoint(step, result.hit_rate)
                    print(f"[Checkpoint] new best hit_rate={result.hit_rate:.2%} -> saved.")
                self.eval_history.append(
                    {
                        "step": step,
                        "hit_rate": result.hit_rate,
                        "epsilon": self.agent.epsilon,
                        "loss": self._last_loss,
                    }
                )
                self.log_diagnostics(step)

        total_elapsed = time.monotonic() - self._start_time
        print(f"[Train] finished {self.config.num_training_steps:,} steps in {_format_duration(total_elapsed)}.")

    def evaluate(self) -> EvaluationResult:
        total_hits = 0
        total_accesses = 0

        for episode in range(self._num_eval_episodes):
            self.eval_environment.reset()
            last_info: dict = {"num_hits": 0, "num_misses": 0}
            decisions = 0

            while not self.eval_environment.done:
                state = self.eval_environment.current_state()
                action = self.agent.select_action(state, greedy=True)
                result = self.eval_environment.step(action)
                last_info = result.info
                decisions += 1
                if decisions % _EVAL_PROGRESS_INTERVAL == 0:
                    running_hit_rate = (
                        last_info["num_hits"]
                        / (last_info["num_hits"] + last_info["num_misses"])
                        if (last_info["num_hits"] + last_info["num_misses"]) > 0
                        else 0.0
                    )
                    print(
                        f"[Eval]   ...{decisions:,} decisions "
                        f"(running hit_rate={running_hit_rate:.2%})"
                    )

            total_hits += last_info["num_hits"]
            total_accesses += last_info["num_hits"] + last_info["num_misses"]

        hit_rate = total_hits / total_accesses if total_accesses > 0 else 0.0
        return EvaluationResult(
            hit_rate=hit_rate,
            num_episodes=self._num_eval_episodes,
            num_hits=total_hits,
            num_misses=total_accesses - total_hits,
        )

    def log(self, step: int) -> None:
        loss_str = f"{self._last_loss:.4f}" if self._last_loss is not None else "n/a"

        avg_reward = (
            self._reward_sum_since_log / self._reward_count_since_log
            if self._reward_count_since_log > 0
            else 0.0
        )
        self._reward_sum_since_log = 0.0
        self._reward_count_since_log = 0

        pct_complete = step / self.config.num_training_steps
        elapsed = time.monotonic() - self._start_time
        steps_per_sec = step / elapsed if elapsed > 0 else 0.0
        remaining_steps = self.config.num_training_steps - step
        eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0.0

        print(
            f"[Train] step {step:>7,}/{self.config.num_training_steps:,} "
            f"({pct_complete:>5.1%}) | "
            f"loss={loss_str} | "
            f"eps={self.agent.epsilon:.3f} | "
            f"buffer={len(self.replay_buffer):,} | "
            f"avg_reward={avg_reward:+.3f} | "
            f"elapsed={_format_duration(elapsed)} | "
            f"eta={_format_duration(eta)}"
        )

    def log_diagnostics(self, step: int) -> None:
        """Stage 4 instrumentation: distributional stats for diagnosing
        value drift, rather than inferring it from hit rate alone.

        Prints two lines:
        - [Diag] return / segment_len: distribution of realized transition
          reward and steps_elapsed since the last call. Directly tests the
          Priority 4 hypothesis that segment length (and therefore return
          magnitude) grows as the policy improves.
        - [Diag] Q / target: distribution of the online network's Q(s,a)
          for the sampled actions and the TD targets computed against
          them, from the most recent train_step() call. A climbing Q mean
          alongside falling hit rate is the direct signature of value
          drift the audit was looking for.

        Read-only: does not affect training. Clears the reward/segment
        buffers so the next window reflects only steps since this call.
        """
        if self._diag_rewards:
            rewards_arr = np.asarray(self._diag_rewards, dtype=np.float64)
            lengths_arr = np.asarray(self._diag_segment_lengths, dtype=np.float64)
            print(
                f"[Diag]  step={step:,} "
                f"return: mean={rewards_arr.mean():+.3f} "
                f"std={rewards_arr.std():.3f} "
                f"max={rewards_arr.max():+.3f} | "
                f"segment_len: mean={lengths_arr.mean():.1f} "
                f"std={lengths_arr.std():.1f} "
                f"max={int(lengths_arr.max())}"
            )
        self._diag_rewards = []
        self._diag_segment_lengths = []

        stats = self.agent.last_batch_stats
        if stats:
            print(
                f"[Diag]  step={step:,} "
                f"Q: mean={stats['q_mean']:+.3f} "
                f"std={stats['q_std']:.3f} "
                f"max={stats['q_max']:+.3f} | "
                f"target: mean={stats['target_mean']:+.3f} "
                f"std={stats['target_std']:.3f} "
                f"max={stats['target_max']:+.3f}"
            )

    def save_checkpoint(self, step: int, hit_rate: float) -> None:
        path = self._checkpoint_dir / "best.pt"
        torch.save(
            {
                "step": step,
                "hit_rate": hit_rate,
                "online_state_dict": self.agent.online_network.state_dict(),
            },
            path,
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
            steps_elapsed=result.steps_elapsed,
        )