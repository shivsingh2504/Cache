from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from RL.config import EnvironmentConfig
from RL.normalizer import StateNormalizer
from feature.extractor import FeatureExtractor
from feature.schema import FeatureVector
from feature.state import CacheState
from lstm.predictor import Predictor
from simulator.generator import WorkloadGenerator


CacheKey = Any


@dataclass
class StepResult:
    next_state: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]
    steps_elapsed: int


class CacheEvictionEnvironment:
    def __init__(
        self,
        config: EnvironmentConfig,
        workload_generator: WorkloadGenerator,
        predictor: Predictor,
        cache_state: CacheState,
        normalizer: StateNormalizer,
    ) -> None:
        self.config = config
        self.workload_generator = workload_generator
        self.predictor = predictor
        self.normalizer = normalizer

        self._cache_state = cache_state
        self._feature_extractor = FeatureExtractor(self._cache_state)

        self._slot_keys: list[CacheKey | None] = [
            None
        ] * self.config.cache_capacity

        self._context_window: deque[FeatureVector] = deque(
            maxlen=self.config.context_window_size
        )

        self._workload_iter = None

        self._pending_miss_key: CacheKey | None = None

        self._num_hits = 0
        self._num_misses = 0

        self._is_done = False

    def reset(self) -> np.ndarray:
        self._cache_state = CacheState()
        self._feature_extractor = FeatureExtractor(self._cache_state)

        self.predictor.set_state(self._cache_state)

        self._slot_keys = [None] * self.config.cache_capacity

        self._context_window.clear()

        self._workload_iter = iter(
            self.workload_generator.generate()
        )

        self._pending_miss_key = None

        self._num_hits = 0
        self._num_misses = 0

        self._is_done = False

        self._advance_to_next_decision()

        return self.current_state()

    def current_state(self) -> np.ndarray:
        state = np.zeros(
            (
                self.config.cache_capacity,
                self.config.num_state_features,
            ),
            dtype=np.float32,
        )

        resident_keys = [
            key
            for key in self._slot_keys
            if key is not None
        ]

        popularity_scores = self.predictor.score(
            feature_window=self._context_window,
            candidate_keys=resident_keys,
        )

        now = self._cache_state.last_event_timestamp()

        for slot_index, key in enumerate(self._slot_keys):
            if key is None:
                state[slot_index, 4] = 1.0
                continue

            frequency = self._cache_state.frequency_of(key)

            last_access = self._cache_state.last_access_of(key)
            first_access = self._cache_state.first_access_of(key)

            recency = (
                0.0
                if last_access is None or now is None
                else float(now - last_access)
            )

            key_age = (
                0.0
                if first_access is None or now is None
                else float(now - first_access)
            )

            predicted_popularity = popularity_scores.get(
                key,
                0.0,
            )

            state[slot_index] = [
                float(frequency),
                recency,
                key_age,
                float(predicted_popularity),
                0.0,
            ]

        return self.normalizer.normalize(state)

    def raw_state(self) -> np.ndarray:
        """Read-only diagnostic accessor: identical to current_state() but
        returns the pre-normalization feature matrix (raw frequency,
        recency, key_age, predicted_popularity, is_empty), so diagnostics
        can inspect actual scale/drift without re-deriving normalize()'s
        inverse. Does not mutate any state and is not used by
        current_state() or step().
        """
        state = np.zeros(
            (
                self.config.cache_capacity,
                self.config.num_state_features,
            ),
            dtype=np.float32,
        )

        resident_keys = [
            key
            for key in self._slot_keys
            if key is not None
        ]

        popularity_scores = self.predictor.score(
            feature_window=self._context_window,
            candidate_keys=resident_keys,
        )

        now = self._cache_state.last_event_timestamp()

        for slot_index, key in enumerate(self._slot_keys):
            if key is None:
                state[slot_index, 4] = 1.0
                continue

            frequency = self._cache_state.frequency_of(key)
            last_access = self._cache_state.last_access_of(key)
            first_access = self._cache_state.first_access_of(key)

            recency = (
                0.0
                if last_access is None or now is None
                else float(now - last_access)
            )
            key_age = (
                0.0
                if first_access is None or now is None
                else float(now - first_access)
            )
            predicted_popularity = popularity_scores.get(key, 0.0)

            state[slot_index] = [
                float(frequency),
                recency,
                key_age,
                float(predicted_popularity),
                0.0,
            ]

        return state

    def step(self, action: int) -> StepResult:
        if not 0 <= action < self.config.cache_capacity:
            raise ValueError(f"action {action} out of range")
        if self._pending_miss_key is None:
            raise RuntimeError("No pending miss to resolve.")

        evicted_key = self._slot_keys[action]
        self._slot_keys[action] = self._pending_miss_key
        self._pending_miss_key = None
        self._num_misses += 1

        advance_result = self._advance_to_next_decision()
        reward = advance_result["reward_delta"]

        return StepResult(
            next_state=self.current_state(),
            reward=reward,
            done=self._is_done,
            info={"num_hits": self._num_hits, "num_misses": self._num_misses, "evicted_key": evicted_key},
            steps_elapsed=advance_result["steps_elapsed"],
        )

    def _advance_to_next_decision(self) -> dict[str, Any]:
        reward_delta = 0.0
        steps_elapsed = 0
        discount = 1.0

        for event in self._workload_iter:
            steps_elapsed += 1
            feature = self._feature_extractor.extract(event)
            self._context_window.append(feature)

            is_hit = event.key in self._slot_keys
            if is_hit:
                self._num_hits += 1
                reward_delta += discount * 1.0
                discount *= self.config.gamma
                continue

            free_slot = next(
                (index for index, key in enumerate(self._slot_keys) if key is None),
                None,
            )
            if free_slot is not None:
                self._slot_keys[free_slot] = event.key
                self._num_misses += 1
                reward_delta -= discount * 1.0
                discount *= self.config.gamma
                continue

            reward_delta -= discount * 1.0
            self._pending_miss_key = event.key
            return {"reward_delta": reward_delta, "steps_elapsed": steps_elapsed}

        self._is_done = True
        return {"reward_delta": reward_delta, "steps_elapsed": steps_elapsed}

    @property
    def done(self) -> bool:
        return self._is_done