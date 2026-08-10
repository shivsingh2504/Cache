"""Run this WHILE your main.py training is still going -- it only needs the
eval workload trace, not the trained agent, so it answers the "is 90%
realistic" question in seconds instead of after a 45-minute wait.

Usage:
    python quick_ceiling_check.py
"""
from __future__ import annotations

from simulator.config import WorkloadConfig, DistributionType
from simulator.generator import WorkloadGenerator

from RL.baseline import FIFOPolicy, LFUPolicy, LRUPolicy
from RL.belady import compute_belady_hit_rate
from RL.config import EnvironmentConfig, NetworkConfig
from RL.environment import CacheEvictionEnvironment
from RL.normalizer import StateNormalizer
from RL.benchmark import evaluate_policy, FixedTraceWorkloadGenerator, _materialize_trace

from feature.state import CacheState
from lstm.config import ModelConfig
from lstm.predictor import Predictor


def main() -> None:
    # Must match main.py's eval_workload_config exactly.
    eval_workload_config = WorkloadConfig(
        num_keys=5000,
        num_requests=20_000,
        distribution=DistributionType.ZIPFIAN,
        seed=43,
    )
    eval_workload = WorkloadGenerator(eval_workload_config)
    trace = _materialize_trace(eval_workload)

    print(f"Trace length: {len(trace):,} events")
    print(f"zipf_alpha: {eval_workload_config.zipf_alpha}")
    print()

    network_config = NetworkConfig()
    environment_config = EnvironmentConfig(cache_capacity=network_config.cache_capacity)

    model_config = ModelConfig(event_features=8, candidate_features=3)
    cache_state = CacheState()
    predictor = Predictor.from_checkpoint(
        checkpoint_path="lstm_popularity_predictor.pt",
        model_config=model_config,
        state=cache_state,
    )
    normalizer = StateNormalizer()

    def make_env() -> CacheEvictionEnvironment:
        return CacheEvictionEnvironment(
            config=environment_config,
            workload_generator=FixedTraceWorkloadGenerator(trace=trace),
            predictor=predictor,
            cache_state=CacheState(),
            normalizer=normalizer,
        )

    print("Evaluating LRU...")
    lru = evaluate_policy(make_env(), LRUPolicy())
    print("Evaluating LFU...")
    lfu = evaluate_policy(make_env(), LFUPolicy())
    print("Evaluating FIFO...")
    fifo = evaluate_policy(make_env(), FIFOPolicy())
    print("Evaluating Belady (offline optimum)...")
    belady = compute_belady_hit_rate(trace=trace, cache_capacity=environment_config.cache_capacity)

    print()
    print("=" * 50)
    print(f"{'Policy':<10}{'Hit Rate':>12}")
    print("-" * 50)
    print(f"{'LRU':<10}{lru.hit_rate:>12.2%}")
    print(f"{'LFU':<10}{lfu.hit_rate:>12.2%}")
    print(f"{'FIFO':<10}{fifo.hit_rate:>12.2%}")
    print(f"{'Belady':<10}{belady.hit_rate:>12.2%}   <-- ceiling, no policy can beat this")
    print("=" * 50)


if __name__ == "__main__":
    main()