from __future__ import annotations

from dataclasses import dataclass

from RL.trainer import EvaluationResult
from simulator.schema import AccessEvent


def _next_occurrence_indices(keys: list[int]) -> list[int | None]:
    n = len(keys)
    next_occurrence: list[int | None] = [None] * n
    last_seen: dict[int, int] = {}
    for i in range(n - 1, -1, -1):
        k = keys[i]
        next_occurrence[i] = last_seen.get(k)
        last_seen[k] = i
    return next_occurrence


def compute_belady_hit_rate(
    trace: list[AccessEvent],
    cache_capacity: int,
) -> EvaluationResult:
    keys = [event.key for event in trace]
    next_occurrence = _next_occurrence_indices(keys)

    cache: set[int] = set()
    last_index_in_cache: dict[int, int] = {}

    num_hits = 0
    num_misses = 0

    for i, key in enumerate(keys):
        if key in cache:
            num_hits += 1
            last_index_in_cache[key] = i
            continue

        num_misses += 1

        if len(cache) < cache_capacity:
            cache.add(key)
            last_index_in_cache[key] = i
            continue

        farthest_key = None
        farthest_distance = -1
        for resident_key in cache:
            occ = next_occurrence[last_index_in_cache[resident_key]]
            distance = float("inf") if occ is None else occ
            if distance > farthest_distance:
                farthest_distance = distance
                farthest_key = resident_key

        cache.remove(farthest_key)
        cache.add(key)
        last_index_in_cache[key] = i

    total = num_hits + num_misses
    hit_rate = num_hits / total if total > 0 else 0.0
    return EvaluationResult(
        hit_rate=hit_rate,
        num_episodes=1,
        num_hits=num_hits,
        num_misses=num_misses,
    )