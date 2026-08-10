from dataclasses import dataclass
import random

import numpy as np

from RL.config import ReplayBufferConfig


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    steps_elapsed: int


@dataclass
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    steps_elapsed: np.ndarray


class ReplayBuffer:
    """Fixed-capacity experience replay using a preallocated ring buffer.

    Backed by a Python list indexed with O(1) random access, rather than a
    collections.deque (whose __getitem__ is O(n)). API is identical to the
    previous deque-backed implementation: push(), sample(), __len__().
    """

    def __init__(self, config: ReplayBufferConfig) -> None:
        self.config = config
        self._buffer: list[Transition | None] = [None] * config.capacity
        self._size = 0
        self._next_index = 0

    def push(self, transition: Transition) -> None:
        self._buffer[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self.config.capacity
        self._size = min(self._size + 1, self.config.capacity)

    def sample(
        self,
        batch_size: int | None = None,
    ) -> TransitionBatch:
        resolved_batch_size = (
            batch_size
            if batch_size is not None
            else self.config.batch_size
        )

        if self._size < resolved_batch_size:
            raise ValueError(
                f"Cannot sample batch_size={resolved_batch_size} "
                f"from a buffer containing only "
                f"{self._size} transitions."
            )

        indices = random.sample(range(self._size), resolved_batch_size)
        sampled = [self._buffer[i] for i in indices]

        return TransitionBatch(
            states=np.stack(
                [t.state for t in sampled],
                axis=0,
            ),
            actions=np.array(
                [t.action for t in sampled],
                dtype=np.int64,
            ),
            rewards=np.array(
                [t.reward for t in sampled],
                dtype=np.float32,
            ),
            next_states=np.stack(
                [t.next_state for t in sampled],
                axis=0,
            ),
            dones=np.array(
                [t.done for t in sampled],
                dtype=np.bool_,
            ),
            steps_elapsed=np.array(
                [t.steps_elapsed for t in sampled],
                dtype=np.int64,
            ),
        )

    def __len__(self) -> int:
        return self._size