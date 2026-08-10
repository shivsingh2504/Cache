from __future__ import annotations

from typing import Protocol

import numpy as np

_FREQUENCY_COL = 0
_RECENCY_COL = 1
_KEY_AGE_COL = 2
_IS_EMPTY_COL = 4


class Policy(Protocol):
    def select_action(self, state: np.ndarray) -> int: ...


class LRUPolicy:

    def select_action(self, state: np.ndarray) -> int:
        recency = np.where(
            state[:, _IS_EMPTY_COL] > 0.5,
            -np.inf,
            state[:, _RECENCY_COL],
        )
        return int(np.argmax(recency))


class LFUPolicy:
    def select_action(self, state: np.ndarray) -> int:
        frequency = np.where(
            state[:, _IS_EMPTY_COL] > 0.5,
            np.inf,
            state[:, _FREQUENCY_COL],
        )
        return int(np.argmin(frequency))


class FIFOPolicy:

    def select_action(self, state: np.ndarray) -> int:
        key_age = np.where(
            state[:, _IS_EMPTY_COL] > 0.5,
            -np.inf,
            state[:, _KEY_AGE_COL],
        )
        return int(np.argmax(key_age))