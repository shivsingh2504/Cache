import random
from collections import deque
from dataclasses import dataclass

from feature.state import CacheState
from lstm.config import DatasetConfig
from model_input.candidate_builder import CandidateFeatureBuilder
from model_input.schema import TrainingSample


class CandidatePool:
    def __init__(self, window_size: int) -> None:
        self._window_size = window_size
        self._recent_keys: deque[int] = deque(maxlen=window_size)

    def observe(self, key: int) -> None:
        self._recent_keys.append(key)

    def sample(self, n: int, rng: random.Random) -> list[int]:
        pool = list(set(self._recent_keys))
        if not pool:
            return []
        n = min(n, len(pool))
        return rng.sample(pool, n)


class LabelGenerator:
    def __init__(self, horizon: int) -> None:
        self._horizon = horizon

    def label_for(self, future_keys: list[int], candidate_key: int) -> float:
        window = future_keys[: self._horizon]
        return float(window.count(candidate_key))


@dataclass
class TrainingSampleBuilder:
    dataset_config: DatasetConfig
    state: CacheState
    candidate_pool: CandidatePool
    label_generator: LabelGenerator
    rng: random.Random

    def __post_init__(self) -> None:
        self._feature_builder = CandidateFeatureBuilder(self.state)

    def build_samples(
        self,
        context: tuple,
        candidates: list[int],
        future_keys: list[int],
    ) -> list[TrainingSample]:
        samples = []
        for key in candidates:
            candidate_feature = self._feature_builder.build(key)
            label = self.label_generator.label_for(future_keys, key)
            samples.append(
                TrainingSample(context=context, candidate=candidate_feature, label=label)
            )
        return samples


@dataclass
class _PendingSample:
    context: tuple
    candidates: list[int]
    captured_at: int  # event index (count of observed keys) when context was captured


class PendingSampleQueue:
    """Buffers (context, candidates) until enough future keys have streamed
    in to compute their labels. Owns the lag between context capture and
    label availability — no other component in the pipeline is responsible
    for this, since WorkloadGenerator is a forward-only stream and labels
    require H events that haven't happened yet at capture time."""

    def __init__(self, horizon: int) -> None:
        self._horizon = horizon
        self._future_keys: deque[int] = deque()
        self._pending: deque[_PendingSample] = deque()
        self._events_seen = 0

    def observe_key(self, key: int) -> None:
        self._future_keys.append(key)
        self._events_seen += 1

    def enqueue(self, context: tuple, candidates: list[int]) -> None:
        self._pending.append(
            _PendingSample(context=context, candidates=candidates, captured_at=self._events_seen)
        )

    def ready_batches(self) -> list[tuple[tuple, list[int], list[int]]]:
        """Returns (context, candidates, future_keys) for every pending
        sample whose horizon has fully elapsed. future_keys is the H keys
        immediately following the sample's capture time."""
        ready = []
        while self._pending and self._pending[0].captured_at + self._horizon <= self._events_seen:
            sample = self._pending.popleft()
            start = sample.captured_at
            future = list(self._future_keys)[start : start + self._horizon]
            ready.append((sample.context, sample.candidates, future))
        return ready