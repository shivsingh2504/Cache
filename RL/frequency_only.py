"""Frequency-only DQN ablation.

Diagnostic experiment only -- not a production component. Answers one
question: can the DQN learn a simple frequency-minimization policy (the
one LFU already implements directly) when given only the information LFU
fundamentally uses?

This module composes existing, untouched pieces (RL/environment.py,
RL/benchmark.py, RL/baseline.py, RL/belady.py, RL/agent.py) rather than
editing any of them.

Verified state layout, from RL/environment.py::
CacheEvictionEnvironment.current_state():

    column 0 = frequency
    column 1 = recency
    column 2 = key_age
    column 3 = predicted_popularity
    column 4 = is_empty

The ablation preserves columns 0 and 4, zeros columns 1-3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from RL.agent import DQNAgent
from RL.baseline import FIFOPolicy, LFUPolicy, LRUPolicy
from RL.belady import compute_belady_hit_rate
from RL.benchmark import (
    ORACLE_POLICY_NAME,
    BenchmarkEntry,
    FixedTraceWorkloadGenerator,
    PolicyKind,
    evaluate_policy,
)
from RL.environment import CacheEvictionEnvironment, StepResult
from feature.state import CacheState

LEARNED_POLICY_NAME_FREQ_ONLY = "Cache Intelligence (Frequency-only DQN)"

# recency, key_age, predicted_popularity -- everything except frequency
# (col 0) and is_empty (col 4).
_MASKED_COLUMNS = (1, 2, 3)


def mask_frequency_only(state: np.ndarray) -> np.ndarray:
    """Returns a COPY of `state` with recency/key_age/predicted_popularity
    zeroed, leaving frequency and is_empty untouched.

    Returns a copy rather than mutating in place: `state` may be a
    reference the caller (the environment, the replay buffer, or another
    consumer) still holds and depends on being unmodified.
    """
    masked = state.copy()
    masked[:, list(_MASKED_COLUMNS)] = 0.0
    return masked


class FrequencyOnlyEnvironment:
    """Wraps an existing CacheEvictionEnvironment and masks every state it
    returns. Delegates everything else unchanged. Does not edit or
    subclass CacheEvictionEnvironment.

    Use for training and periodic in-training evaluation only. NOT used
    for the final benchmark's DQN row -- run_baseline_comparison()
    rebuilds a fresh raw CacheEvictionEnvironment internally from
    attribute access, which would silently bypass this wrapper for the
    learned-policy row. The final benchmark instead applies
    mask_frequency_only() at the policy level via
    FrequencyOnlyGreedyAgentPolicy, so baselines stay on the normal,
    unmasked, unmodified path.
    """

    def __init__(self, environment: CacheEvictionEnvironment) -> None:
        self._environment = environment

    def reset(self) -> np.ndarray:
        state = self._environment.reset()
        return mask_frequency_only(state)

    def current_state(self) -> np.ndarray:
        state = self._environment.current_state()
        return mask_frequency_only(state)

    def step(self, action: int) -> StepResult:
        result = self._environment.step(action)
        return StepResult(
            next_state=mask_frequency_only(result.next_state),
            reward=result.reward,
            done=result.done,
            info=result.info,
            steps_elapsed=result.steps_elapsed,
        )

    @property
    def done(self) -> bool:
        return self._environment.done

    # Pass-throughs. Trainer.warmup() reads .config.cache_capacity;
    # nothing else on this wrapper is read by Trainer. Included for
    # completeness / future callers, not because Trainer currently needs
    # all of them.
    @property
    def config(self):
        return self._environment.config

    @property
    def workload_generator(self):
        return self._environment.workload_generator

    @property
    def predictor(self):
        return self._environment.predictor

    @property
    def normalizer(self):
        return self._environment.normalizer


@dataclass
class FrequencyOnlyGreedyAgentPolicy:
    """Policy-level counterpart to benchmark.py's GreedyAgentPolicy. Masks
    the state immediately before inference so the DQN sees exactly what it
    was trained on, while the environment supplying that state remains the
    normal, unmasked CacheEvictionEnvironment shared with every other
    benchmarked policy.
    """

    agent: DQNAgent

    def select_action(self, state: np.ndarray) -> int:
        masked_state = mask_frequency_only(state)
        return self.agent.select_action(masked_state, greedy=True)


def run_frequency_only_benchmark(
    environment: CacheEvictionEnvironment,
    agent: DQNAgent,
    num_episodes: int = 1,
) -> list[BenchmarkEntry]:
    """Mirrors RL.benchmark.run_baseline_comparison's methodology exactly
    -- same trace materialization, same evaluate_policy/
    compute_belady_hit_rate calls, same fresh-environment-per-policy
    construction -- except the learned-policy row uses
    FrequencyOnlyGreedyAgentPolicy instead of GreedyAgentPolicy.
    FIFO/LRU/LFU/Belady are evaluated completely unmasked, via the same
    functions benchmark.py itself uses, imported and reused here rather
    than reimplemented.

    `environment` must be a FRESH, held-out, RAW (unwrapped)
    CacheEvictionEnvironment -- do not pass a FrequencyOnlyEnvironment
    here; masking for the DQN row happens at the policy level instead so
    the baselines can share the identical unmasked construction path.
    """
    trace = list(environment.workload_generator.generate())
    cache_capacity = environment.config.cache_capacity

    def make_benchmark_environment() -> CacheEvictionEnvironment:
        return CacheEvictionEnvironment(
            config=environment.config,
            workload_generator=FixedTraceWorkloadGenerator(trace=trace),
            predictor=environment.predictor,
            cache_state=CacheState(),
            normalizer=environment.normalizer,
        )

    entries: list[BenchmarkEntry] = []

    for name, policy in (
        ("FIFO", FIFOPolicy()),
        ("LRU", LRUPolicy()),
        ("LFU", LFUPolicy()),
    ):
        print(f"[Benchmark] evaluating {name}...")
        result = evaluate_policy(make_benchmark_environment(), policy, num_episodes)
        entries.append(BenchmarkEntry(name=name, kind=PolicyKind.HEURISTIC, result=result))

    print(f"[Benchmark] evaluating {LEARNED_POLICY_NAME_FREQ_ONLY}...")
    freq_only_result = evaluate_policy(
        make_benchmark_environment(),
        FrequencyOnlyGreedyAgentPolicy(agent),
        num_episodes,
    )
    entries.append(
        BenchmarkEntry(
            name=LEARNED_POLICY_NAME_FREQ_ONLY,
            kind=PolicyKind.LEARNED,
            result=freq_only_result,
        )
    )

    print(f"[Benchmark] evaluating {ORACLE_POLICY_NAME}...")
    belady_result = compute_belady_hit_rate(trace=trace, cache_capacity=cache_capacity)
    entries.append(BenchmarkEntry(name=ORACLE_POLICY_NAME, kind=PolicyKind.ORACLE, result=belady_result))

    environment.reset()
    return entries