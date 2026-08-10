from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator

import numpy as np

from RL.agent import DQNAgent
from RL.baseline import FIFOPolicy, LFUPolicy, LRUPolicy, Policy
from RL.belady import compute_belady_hit_rate
from RL.environment import CacheEvictionEnvironment
from RL.trainer import EvaluationResult
from feature.state import CacheState
from simulator.generator import WorkloadGenerator
# AccessEvent is defined in simulator.schema; simulator.generator previously
# re-exported the same name, which shadowed this import and risked pulling
# in the wrong type if the two ever diverged. schema.py is the canonical
# definition site, so it is imported from there exclusively.
from simulator.schema import AccessEvent


# Display name for the learned policy. Single source of truth -- used both
# when registering the policy for evaluation and when the printer decides
# how to label/group it. Change it here and it propagates everywhere.
LEARNED_POLICY_NAME = "Cache Intelligence (LSTM + DQN)"
ORACLE_POLICY_NAME = "Belady (Oracle)"


class PolicyKind(Enum):
    """Classifies a benchmark entry so the printer can group and compare
    correctly without hardcoding policy names anywhere except the single
    registry built in run_baseline_comparison().

    HEURISTIC: a deployable, classical eviction rule (FIFO/LRU/LFU).
    LEARNED:   the trained system under evaluation.
    ORACLE:    an offline-optimal upper bound (Belady/MIN). Requires full
               future knowledge of the trace and is not a real policy --
               excluded from "best policy" comparisons, used only to
               measure headroom.
    """

    HEURISTIC = "heuristic"
    LEARNED = "learned"
    ORACLE = "oracle"


@dataclass(frozen=True)
class BenchmarkEntry:
    name: str
    kind: PolicyKind
    result: EvaluationResult


@dataclass(frozen=True)
class PolicySpec:
    """One row of the evaluation registry. `evaluate` is a zero-arg
    closure that already has its environment/trace/agent bound, so the
    runner loop below is identical for every kind of policy -- heuristic,
    learned, or oracle -- and adding a new policy never requires touching
    the runner or the printer."""

    name: str
    kind: PolicyKind
    evaluate: Callable[[], EvaluationResult]


@dataclass
class GreedyAgentPolicy:
    agent: DQNAgent

    def select_action(self, state: np.ndarray) -> int:
        return self.agent.select_action(state, greedy=True)


@dataclass
class FixedTraceWorkloadGenerator:
    trace: list[AccessEvent]

    def generate(self) -> Iterator[AccessEvent]:
        return iter(self.trace)


def _materialize_trace(generator: WorkloadGenerator) -> list[AccessEvent]:
    return list(generator.generate())


def evaluate_policy(
    environment: CacheEvictionEnvironment,
    policy: Policy,
    num_episodes: int = 1,
) -> EvaluationResult:
    total_hits = 0
    total_accesses = 0

    for _ in range(num_episodes):
        environment.reset()
        last_info: dict = {"num_hits": 0, "num_misses": 0}

        while not environment.done:
            state = environment.current_state()
            action = policy.select_action(state)
            result = environment.step(action)
            last_info = result.info

        total_hits += last_info["num_hits"]
        total_accesses += last_info["num_hits"] + last_info["num_misses"]

    hit_rate = total_hits / total_accesses if total_accesses > 0 else 0.0
    return EvaluationResult(
        hit_rate=hit_rate,
        num_episodes=num_episodes,
        num_hits=total_hits,
        num_misses=total_accesses - total_hits,
    )


def _build_policy_registry(
    make_benchmark_environment: Callable[[], CacheEvictionEnvironment],
    trace: list[AccessEvent],
    cache_capacity: int,
    agent: DQNAgent,
    num_episodes: int,
) -> list[PolicySpec]:
    """Single registry that both the evaluator and the printer read from.
    Order here is display order: FIFO -> LRU -> LFU -> learned system ->
    oracle. Every entry is evaluated identically (call `.evaluate()`);
    heuristics and the learned policy share the same online Policy
    protocol via evaluate_policy(), while Belady needs the raw trace and
    key identity (not exposed in `state`), so its closure calls the
    dedicated oracle function instead. That difference lives here, once,
    not scattered through the runner or printer.
    """
    return [
        PolicySpec(
            name="FIFO",
            kind=PolicyKind.HEURISTIC,
            evaluate=lambda: evaluate_policy(
                make_benchmark_environment(), FIFOPolicy(), num_episodes
            ),
        ),
        PolicySpec(
            name="LRU",
            kind=PolicyKind.HEURISTIC,
            evaluate=lambda: evaluate_policy(
                make_benchmark_environment(), LRUPolicy(), num_episodes
            ),
        ),
        PolicySpec(
            name="LFU",
            kind=PolicyKind.HEURISTIC,
            evaluate=lambda: evaluate_policy(
                make_benchmark_environment(), LFUPolicy(), num_episodes
            ),
        ),
        PolicySpec(
            name=LEARNED_POLICY_NAME,
            kind=PolicyKind.LEARNED,
            evaluate=lambda: evaluate_policy(
                make_benchmark_environment(), GreedyAgentPolicy(agent), num_episodes
            ),
        ),
        PolicySpec(
            name=ORACLE_POLICY_NAME,
            kind=PolicyKind.ORACLE,
            evaluate=lambda: compute_belady_hit_rate(
                trace=trace, cache_capacity=cache_capacity
            ),
        ),
    ]


def run_baseline_comparison(
    environment: CacheEvictionEnvironment,
    agent: DQNAgent,
    num_episodes: int = 1,
) -> list[BenchmarkEntry]:
    """Evaluates every registered policy -- heuristic, learned, and oracle
    -- on the exact same fixed, replayed trace.

    `environment` must be a held-out evaluation environment (never the
    training environment), so the comparison reflects generalization
    rather than performance on data the agent has already trained on.

    Returns an ordered list of BenchmarkEntry rather than a bare dict so
    each result carries its PolicyKind for downstream printing/analysis
    without re-deriving it from the name string.
    """
    trace = _materialize_trace(environment.workload_generator)
    cache_capacity = environment.config.cache_capacity

    def make_benchmark_environment() -> CacheEvictionEnvironment:
        return CacheEvictionEnvironment(
            config=environment.config,
            workload_generator=FixedTraceWorkloadGenerator(trace=trace),
            predictor=environment.predictor,
            cache_state=CacheState(),
            normalizer=environment.normalizer,
        )

    registry = _build_policy_registry(
        make_benchmark_environment=make_benchmark_environment,
        trace=trace,
        cache_capacity=cache_capacity,
        agent=agent,
        num_episodes=num_episodes,
    )

    entries: list[BenchmarkEntry] = []
    for spec in registry:
        print(f"[Benchmark] evaluating {spec.name}...")
        entries.append(BenchmarkEntry(name=spec.name, kind=spec.kind, result=spec.evaluate()))

    environment.reset()
    return entries


def _pct_improvement(value: float, over: float) -> str:
    if over <= 0:
        return "\u2014"  # em dash
    return f"{(value - over) / over * 100:+.1f}%"


def print_comparison(entries: list[BenchmarkEntry]) -> None:
    """Prints a research-paper-style benchmark table: heuristics and the
    learned system compared directly, the oracle set apart as an upper
    bound, plus a derived-statistics summary (gap to oracle, improvement
    over baselines, headroom closed).
    """
    by_name = {entry.name: entry for entry in entries}
    deployable = [e for e in entries if e.kind != PolicyKind.ORACLE]
    oracle = next((e for e in entries if e.kind == PolicyKind.ORACLE), None)
    learned = next((e for e in entries if e.kind == PolicyKind.LEARNED), None)
    heuristics = [e for e in entries if e.kind == PolicyKind.HEURISTIC]

    lru = by_name.get("LRU")
    fifo = by_name.get("FIFO")

    width = 72
    print("=" * width)
    print("CACHE INTELLIGENCE SYSTEM \u2014 BENCHMARK RESULTS")
    print("=" * width)
    print()
    print(
        f"{'Policy':<34}{'Hit Rate':>10}{'vs LRU':>14}{'vs FIFO':>14}"
    )
    print("-" * width)

    for entry in deployable:
        vs_lru = (
            "\u2014"
            if lru is None or entry.name == "LRU"
            else _pct_improvement(entry.result.hit_rate, lru.result.hit_rate)
        )
        vs_fifo = (
            "\u2014"
            if fifo is None or entry.name == "FIFO"
            else _pct_improvement(entry.result.hit_rate, fifo.result.hit_rate)
        )
        label = entry.name
        if entry.kind == PolicyKind.LEARNED:
            label += "  *"
        print(
            f"{label:<34}{entry.result.hit_rate:>9.2%} "
            f"{vs_lru:>13} {vs_fifo:>14}"
        )

    if oracle is not None:
        print("-" * width)
        print(
            f"{oracle.name + ' (upper bound, not competing)':<34}"
            f"{oracle.result.hit_rate:>9.2%}"
        )

    print("=" * width)
    if learned is not None:
        print("* = learned system (LSTM popularity predictor + DQN eviction policy)")
    print()

    _print_summary(deployable, heuristics, learned, oracle)


def _print_summary(
    deployable: list[BenchmarkEntry],
    heuristics: list[BenchmarkEntry],
    learned: BenchmarkEntry | None,
    oracle: BenchmarkEntry | None,
) -> None:
    print("Summary")
    print("-" * 30)

    best_heuristic = max(heuristics, key=lambda e: e.result.hit_rate) if heuristics else None
    if best_heuristic is not None:
        print(
            f"Best heuristic baseline: {best_heuristic.name} "
            f"({best_heuristic.result.hit_rate:.2%})"
        )

    if learned is not None:
        print(f"{learned.name}: {learned.result.hit_rate:.2%} hit rate")

        if best_heuristic is not None:
            beats = learned.result.hit_rate > best_heuristic.result.hit_rate
            delta_pp = (learned.result.hit_rate - best_heuristic.result.hit_rate) * 100
            verdict = "outperforms" if beats else "does not outperform"
            print(
                f"  -> {verdict} the best heuristic baseline "
                f"({best_heuristic.name}) by {abs(delta_pp):.2f} percentage points"
            )

        if oracle is not None:
            gap_pp = (oracle.result.hit_rate - learned.result.hit_rate) * 100
            print(
                f"  -> Gap to oracle ({oracle.name}): {gap_pp:.2f} percentage points"
            )

            if best_heuristic is not None:
                headroom = oracle.result.hit_rate - best_heuristic.result.hit_rate
                if headroom > 0:
                    closed = (learned.result.hit_rate - best_heuristic.result.hit_rate) / headroom * 100
                    print(
                        f"  -> Closed {closed:.1f}% of the headroom between the best "
                        f"heuristic and the oracle ceiling"
                    )
                else:
                    print(
                        "  -> Best heuristic already matches the oracle on this trace "
                        "(no headroom to close)"
                    )

    print("=" * 30)
    print()